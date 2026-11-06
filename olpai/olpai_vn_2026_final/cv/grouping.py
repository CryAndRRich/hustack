from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn

from common.vision import build_model
from cv.augment import GpuAugment
from cv.config import CLASSES
from cv.data import load_batch

FEATURE_HW = (240, 320)


def _strip_head(model: nn.Module) -> nn.Module:
    name = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)][-1]
    parent, _, attr = name.rpartition(".")
    setattr(model.get_submodule(parent), attr, nn.Identity())
    return model


@torch.inference_mode()
def extract_features(cfg, array: np.ndarray, log=print) -> np.ndarray:
    batch_size = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _strip_head(build_model(cfg.group_arch, num_classes=1, pretrained=True,
                                    pretrained_dir=cfg.pretrained_dir, log=log)).to(device).eval()
    transform = GpuAugment(FEATURE_HW, device, train=False, channels_last=False)
    use_amp = device.type == "cuda"
    out = []
    for start in range(0, len(array), batch_size):
        x = transform(load_batch(array, np.arange(start, min(start + batch_size, len(array))), use_amp))
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            f = model(x).float() + model(x.flip(3)).float()
        out.append(nn.functional.normalize(f, dim=1).cpu())
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return torch.cat(out).numpy()


def build_groups(cfg, names: list[str], labels: np.ndarray, array: np.ndarray, log=print) -> np.ndarray:
    if cfg.split_mode == "random":
        return np.arange(len(names))

    key = hashlib.sha1(json.dumps([names, cfg.group_arch, cfg.versions_per_image]).encode()).hexdigest()[:12]
    path = Path(cfg.cache_dir) / f"groups_{key}.npy"
    if path.exists():
        groups = np.load(path)
        log(f"[groups] reusing {path} ({len(np.unique(groups))} groups)")
        return groups

    from sklearn.cluster import AgglomerativeClustering

    features = extract_features(cfg, array, log)
    groups = np.empty(len(names), dtype=np.int64)
    offset = 0
    sizes = Counter()
    for c in range(len(CLASSES)):
        idx = np.flatnonzero(labels == c)
        if len(idx) == 0:
            continue
        n_clusters = max(1, min(len(idx), round(len(idx) / cfg.versions_per_image)))
        if n_clusters == 1:
            assigned = np.zeros(len(idx), dtype=np.int64)
        else:
            assigned = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward").fit_predict(features[idx])
        groups[idx] = offset + assigned
        offset += n_clusters
        sizes.update(Counter(np.bincount(assigned)))
    np.save(path, groups)
    total = sum(sizes.values())
    log(f"[groups] {offset} groups from {len(names)} images ({cfg.group_arch}, per-class Ward) | group sizes: "
        f"{dict(sorted(sizes.items()))} | exactly {cfg.versions_per_image}: "
        f"{sizes[cfg.versions_per_image] / max(1, total):.0%} of groups")
    return groups


def group_split(labels: np.ndarray, groups: np.ndarray, val_size: float, seed: int):
    rng = np.random.default_rng(seed)
    val = []
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        uniq = rng.permutation(np.unique(groups[idx]))
        target = val_size * len(idx)
        taken = 0
        for g in uniq:
            if taken >= target:
                break
            members = idx[groups[idx] == g]
            val.extend(members)
            taken += len(members)
    val_pos = np.sort(np.asarray(val, dtype=np.int64))
    train_pos = np.setdiff1d(np.arange(len(labels)), val_pos)
    return train_pos, val_pos
