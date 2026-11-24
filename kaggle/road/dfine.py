import hashlib
import json
import sys
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

COMMIT = "956d1709314c2c6a4df6f34de232054578a7449f"
REPO_URL = f"https://github.com/Peterande/D-FINE/archive/{COMMIT}.tar.gz"
WEIGHTS_URL = "https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_l_coco.pth"
CHECKPOINTS = ["best_stg2.pth", "best_stg1.pth", "last.pth"]


def sha16(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def setup(cfg):
    root = Path(cfg.dfine_dir)
    repo = root / f"D-FINE-{COMMIT}"
    if not repo.exists():
        root.mkdir(parents=True, exist_ok=True)
        archive = root / "repo.tar.gz"
        urllib.request.urlretrieve(REPO_URL, archive)
        with tarfile.open(archive) as t:
            t.extractall(root)
    weights = root / "dfine_l_coco.pth"
    if not weights.exists():
        urllib.request.urlretrieve(WEIGHTS_URL, weights)
    print(f"dfine: repo {repo.name}, COCO weights {weights.stat().st_size / 2**20:.0f} MB sha {sha16(weights)}")
    return repo, weights


def write_config(cfg, repo, data, out_dir):
    stop = max(1, cfg.dfine_epochs - cfg.dfine_plain_epochs)
    lr = cfg.dfine_lr
    conf = {
        "__include__": ["../../dataset/custom_detection.yml", "../../runtime.yml", "../include/dataloader.yml",
                        "../include/optimizer.yml", "../include/dfine_hgnetv2.yml"],
        "output_dir": str(out_dir), "num_classes": 4, "remap_mscoco_category": False, "print_freq": 200,
        "checkpoint_freq": 100000, "epochs": cfg.dfine_epochs,
        "HGNetv2": {"name": "B4", "return_idx": [1, 2, 3], "freeze_stem_only": True, "freeze_at": 0,
                    "freeze_norm": True, "pretrained": False},
        "optimizer": {"type": "AdamW", "lr": lr, "betas": [0.9, 0.999], "weight_decay": 0.000125,
                      "params": [{"params": "^(?=.*backbone)(?!.*norm|bn).*$", "lr": lr / 20},
                                 {"params": "^(?=.*(?:encoder|decoder))(?=.*(?:norm|bn)).*$", "weight_decay": 0.0}]},
        "train_dataloader": {"total_batch_size": cfg.dfine_batch, "num_workers": cfg.dfine_workers,
                             "dataset": {"img_folder": str(data / "images" / "train"),
                                         "ann_file": str(data / "train.json"),
                                         "transforms": {"policy": {"epoch": stop}}},
                             "collate_fn": {"stop_epoch": stop, "ema_restart_decay": 0.9999, "base_size_repeat": 4}},
        "val_dataloader": {"total_batch_size": 16, "num_workers": 2,
                           "dataset": {"img_folder": str(data / "images" / "val"), "ann_file": str(data / "val.json")}},
    }
    path = repo / "configs" / "dfine" / "custom" / f"road_{out_dir.name}.yml"
    path.write_text(json.dumps(conf, indent=1))
    return path


def export(out_dir, target):
    for name in CHECKPOINTS:
        p = out_dir / name
        if not p.exists():
            continue
        try:
            state = torch.load(p, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"dfine: cannot read {p}: {e!r}")
            continue
        module = state["ema"]["module"] if "ema" in state else state["model"]
        torch.save(module, target)
        print(f"dfine: exported {name} (epoch {state.get('last_epoch')}) -> {target}")
        return name
    raise FileNotFoundError(f"no readable D-FINE checkpoint in {out_dir}")


def epoch_log(out_dir):
    p = out_dir / "log.txt"
    if not p.exists():
        return []
    rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    return [{"epoch": r["epoch"], "AP": r["test_coco_eval_bbox"][0], "AP50": r["test_coco_eval_bbox"][1],
             "loss": r.get("train_loss")} for r in rows]


class DFine:
    def __init__(self, repo, config, weights, max_det):
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        from src.core import YAMLConfig
        cfg = YAMLConfig(str(config))
        cfg.model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
        self.model = cfg.model.deploy().cuda().eval()
        self.post = cfg.postprocessor.deploy()
        self.size = int(cfg.yaml_cfg["eval_spatial_size"][0])
        self.max_det = max_det

    @torch.inference_mode()
    def __call__(self, image, flip):
        h, w = image.shape[:2]
        t = torch.from_numpy(np.array(image)).cuda().permute(2, 0, 1)[None].float().div_(255)
        t = F.interpolate(t, size=(self.size, self.size), mode="bilinear", align_corners=False, antialias=True)
        if flip:
            t = torch.cat([t, t.flip(3)])
        with torch.autocast("cuda", dtype=torch.float16):
            out = self.model(t)
        out = {k: out[k].float() for k in ("pred_logits", "pred_boxes")}
        sizes = torch.tensor([[w, h]], device=t.device).repeat(len(t), 1)
        labels, boxes, scores = self.post(out, sizes)
        res = []
        for k in range(len(t)):
            b = boxes[k, :self.max_det].cpu().numpy().astype(np.float64)
            if k == 1:
                b[:, [0, 2]] = w - b[:, [2, 0]]
            b[:, [0, 2]] = np.clip(b[:, [0, 2]], 0, w)
            b[:, [1, 3]] = np.clip(b[:, [1, 3]], 0, h)
            d = np.column_stack([labels[k, :self.max_det].cpu().numpy(), scores[k, :self.max_det].cpu().numpy(), b])
            res.append(d[(d[:, 4] - d[:, 2] >= 0.05) & (d[:, 5] - d[:, 3] >= 0.05)])
        return res
