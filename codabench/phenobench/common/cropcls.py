import math
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from scipy.stats import rankdata

from common.data import I, PID, SIZE, X0, X1, Y0, Y1, colour_jitter, extract, morph, resize
from common.models import MEAN, STD
from common.train import seed_worker


def crop_labels(plant, sem):
    p = plant.astype(np.int64)
    v = np.bincount((p * 5 + sem.astype(np.int64)).ravel(), minlength=(int(p.max()) + 1) * 5).reshape(-1, 5)
    return v[:, 1] + v[:, 3] >= v[:, 2] + v[:, 4], v.sum(1) > 0


def save_labels(cache_dir, cache, log):
    path = cache_dir / "plant_crop.npy"
    if path.exists():
        return
    boxes = np.load(cache_dir / "plant_boxes.npy")
    lab = np.zeros(len(boxes), bool)
    for i in np.unique(boxes[:, I]):
        crop, _ = crop_labels(np.asarray(cache["plant_train"][i]), np.asarray(cache["sem_train"][i]))
        rows = np.flatnonzero(boxes[:, I] == i)
        lab[rows] = crop[boxes[rows, PID]]
    np.save(path, lab)
    log(f"crop labels: {len(lab)} train plants, {lab.mean():.1%} crop")


def window(cfg, y0, x0, y1, x1):
    side = int(min(max(cfg.cls_context * max(y1 - y0, x1 - x0), cfg.cls_min_side), SIZE))
    return (y0 + y1) // 2 - side // 2, (x0 + x1) // 2 - side // 2, side


def sample(cfg, img, mask, side):
    x = (img.astype(np.float32) * np.where(mask, 1.0, cfg.cls_dim)[..., None]).astype(np.uint8)
    extra = np.array([math.log(int(mask.sum()) + 1) / 10, math.log(side) / 7], np.float32)
    return resize(x, cfg.cls_size, False), extra


class PlantSet(torch.utils.data.Dataset):
    def __init__(self, cfg, cache_dir):
        self.cfg, self.cache_dir, self.mm = cfg, cache_dir, None
        self.boxes = np.load(cache_dir / "plant_boxes.npy")
        self.labels = np.load(cache_dir / "plant_crop.npy")

    def __len__(self):
        return len(self.boxes)

    def __getitem__(self, n):
        cfg = self.cfg
        if self.mm is None:
            self.mm = {k: np.load(self.cache_dir / f"{k}_train.npy", mmap_mode="r") for k in ("img", "plant")}
        row = self.boxes[n]
        top, left, side = window(cfg, row[Y0], row[X0], row[Y1], row[X1])
        img = extract(self.mm["img"][row[I]], top, left, side)
        m = extract(self.mm["plant"][row[I]], top, left, side) == row[PID]
        j = morph(m, random.randint(-cfg.cls_jitter, cfg.cls_jitter))
        m = j if j.any() else m
        k = random.randrange(4)
        img, m = np.rot90(img, k), np.rot90(m, k)
        if random.random() < 0.5:
            img, m = img[:, ::-1], m[:, ::-1]
        x, extra = sample(cfg, colour_jitter(img.astype(np.float32)), m, side)
        return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))), torch.from_numpy(extra), float(self.labels[n])


class CropClassifier(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        h = backbone.config.hidden_size
        self.head = torch.nn.Sequential(torch.nn.Linear(2 * h + 2, 256), torch.nn.GELU(), torch.nn.Linear(256, 1))

    def forward(self, x, extra):
        t = self.backbone(pixel_values=x).last_hidden_state
        return self.head(torch.cat([t[:, 0], t[:, 1:].mean(1), extra], 1))[:, 0]


def build(cfg):
    from transformers import AutoModel
    return CropClassifier(AutoModel.from_pretrained(cfg.cls_backbone, dtype=torch.float32))


def train_cls(cfg, model, dataset, dev, smoke, log):
    dl = torch.utils.data.DataLoader(dataset, batch_size=cfg.cls_batch, shuffle=True, num_workers=cfg.num_workers,
                                     worker_init_fn=seed_worker, drop_last=True, pin_memory=True,
                                     persistent_workers=cfg.num_workers > 0)
    groups = [{"params": list(model.backbone.parameters()), "lr": cfg.cls_lr},
              {"params": list(model.head.parameters()), "lr": cfg.cls_head_lr}]
    base = [g["lr"] for g in groups]
    opt = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)
    params = [p for g in groups for p in g["params"]]
    scaler = torch.amp.GradScaler("cuda")
    mean, std = MEAN.to(dev), STD.to(dev)
    budget = max(cfg.cls_hours * 3600, 1.0)
    log(f"crop classifier {cfg.cls_backbone}: {len(dataset)} train plants ({dataset.labels.mean():.1%} crop), "
        f"batch {cfg.cls_batch}, input {cfg.cls_size}, {cfg.cls_hours:.2f} h")
    t0, step, losses, done = time.time(), 0, [], False
    model.train()
    while not done:
        for x, extra, y in dl:
            frac = min(step / cfg.smoke_steps, 1.0) if smoke else min((time.time() - t0) / budget, 1.0)
            f = min(1.0, (step + 1) / cfg.warmup_steps) * 0.5 * (1 + math.cos(math.pi * frac))
            for g, b in zip(opt.param_groups, base):
                g["lr"] = b * f
            x = (x.to(dev, non_blocking=True).float() - mean) / std
            with torch.autocast("cuda", dtype=torch.float16):
                logit = model(x, extra.to(dev))
            loss = F.binary_cross_entropy_with_logits(logit.float(), y.to(dev).float())
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, cfg.cls_clip)
            scaler.step(opt)
            scaler.update()
            step += 1
            losses.append(loss.item())
            if step % 200 == 0 or smoke:
                log(f"cls step {step} loss {np.mean(losses[-200:]):.4f} {(time.time() - t0) / step:.2f} s/step "
                    f"lr x{f:.3f} mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
            if (smoke and step >= cfg.smoke_steps) or (not smoke and time.time() - t0 >= budget):
                done = True
                break
    model.eval()
    return {"steps": step, "hours": (time.time() - t0) / 3600, "last_loss": float(np.mean(losses[-200:])),
            "max_mem_gb": torch.cuda.max_memory_allocated() / 2**30}


@torch.no_grad()
def crop_probs(cfg, models, img, plant_map, dev):
    pm = plant_map.astype(np.int32)
    out = np.ones(int(pm.max()), np.float32)
    xs, ex, pids = [], [], []
    for pid, sl in enumerate(ndimage.find_objects(pm), 1):
        if sl is None:
            continue
        top, left, side = window(cfg, sl[0].start, sl[1].start, sl[0].stop, sl[1].stop)
        x, e = sample(cfg, extract(img, top, left, side), extract(pm, top, left, side) == pid, side)
        xs.append(x)
        ex.append(e)
        pids.append(pid)
    mean, std = MEAN.to(dev), STD.to(dev)
    for a in range(0, len(xs), cfg.cls_batch):
        x = (torch.from_numpy(np.stack(xs[a:a + cfg.cls_batch])).to(dev).permute(0, 3, 1, 2).float() - mean) / std
        e = torch.from_numpy(np.stack(ex[a:a + cfg.cls_batch])).to(dev)
        with torch.autocast("cuda", dtype=torch.float16):
            p = torch.stack([torch.sigmoid(m(x, e).float()) for m in models]).mean(0)
        out[np.array(pids[a:a + cfg.cls_batch]) - 1] = p.cpu().numpy()
    return out


def val_check(cfg, models, cache, idx, dev):
    ps, ys = [], []
    for i in idx:
        pm = np.asarray(cache["plant_val"][i]).astype(np.int32)
        if pm.max() == 0:
            continue
        crop, present = crop_labels(pm, np.asarray(cache["sem_val"][i]))
        p = crop_probs(cfg, models, np.asarray(cache["img_val"][i]), pm, dev)
        ps.append(p[present[1:]])
        ys.append(crop[1:][present[1:]])
    p, y = np.concatenate(ps), np.concatenate(ys)
    n1, n0 = int(y.sum()), int((~y).sum())
    auc = float((rankdata(p)[y].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1))
    return {"plants": len(y), "crop": n1, "acc": float(((p >= 0.5) == y).mean()), "auc": auc}
