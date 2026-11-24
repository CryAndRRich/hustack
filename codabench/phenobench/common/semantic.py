import math
import random

import numpy as np
import torch
import torch.nn.functional as F

from common.data import I, SIZE, X0, X1, Y0, Y1, blur, colour_jitter, resize, rot_flip, to_tensors, window_start
from common.models import MEAN, STD, loss_in_fp32
from common.plant import image_tensor, tile_starts
from common.sem_metric import MERGE, confusion, ious

LABELS = {0: "soil", 1: "crop", 2: "weed"}


def load_init(path):
    from transformers import Mask2FormerForUniversalSegmentation
    model = Mask2FormerForUniversalSegmentation.from_pretrained(
        path, id2label=LABELS, label2id={v: k for k, v in LABELS.items()}, ignore_mismatched_sizes=True,
        dtype=torch.float32)
    assert model.config.num_labels == 3, model.config.num_labels
    with torch.no_grad():
        model.criterion.empty_weight.copy_(torch.tensor([1.0, 1.0, 1.0, model.config.no_object_weight]))
    model.loss_fp32 = loss_in_fp32(model)
    return model


class SemSet(torch.utils.data.Dataset):
    def __init__(self, cfg, cache_dir):
        self.cfg, self.cache_dir, self.mm = cfg, cache_dir, None
        boxes = np.load(cache_dir / "plant_boxes.npy")
        self.weeds = boxes[~np.load(cache_dir / "plant_crop.npy")]
        self.n = int(boxes[:, I].max()) + 1
        self.max_instances = 3

    def __len__(self):
        return self.n

    def __getitem__(self, n):
        cfg, out = self.cfg, self.cfg.m2f_size
        if self.mm is None:
            self.mm = {k: np.load(self.cache_dir / f"{k}_train.npy", mmap_mode="r") for k in ("img", "sem")}
        z = math.exp(random.uniform(math.log(cfg.sem_zoom_min), math.log(cfg.sem_zoom_max)))
        w = min(SIZE, round(out / z))
        if random.random() < cfg.sem_weed_p:
            row = self.weeds[random.randrange(len(self.weeds))]
            n, y, x = row[I], window_start(row[Y0], row[Y1], w), window_start(row[X0], row[X1], w)
        else:
            y, x = random.randint(0, SIZE - w), random.randint(0, SIZE - w)
        img = np.array(self.mm["img"][n, y:y + w, x:x + w])
        lab = MERGE[np.asarray(self.mm["sem"][n, y:y + w, x:x + w])] + 1
        img, lab = rot_flip(img, lab)
        img, lab = resize(img, out, False), resize(lab, out, True)
        img = blur(np.ascontiguousarray(img), cfg.blur_p)
        return to_tensors(colour_jitter(img.astype(np.float32)), lab, cfg.m2f_stride)


@torch.no_grad()
def seg(model, x):
    with torch.autocast("cuda", dtype=torch.float16):
        out = model(pixel_values=(x - MEAN.to(x.device)) / STD.to(x.device))
    cls = out.class_queries_logits.float().softmax(-1)[..., :-1]
    return torch.einsum("bqc,bqhw->bchw", cls, out.masks_queries_logits.float().sigmoid())


def up(p, size):
    return F.interpolate(p[None], size=(size, size), mode="bilinear", align_corners=False)[0]


def norm(p):
    return (p / p.sum(0, keepdim=True).clamp(min=1e-6)).half()


@torch.no_grad()
def views(cfg, model, x, tiles=True):
    xs = F.interpolate(x, scale_factor=cfg.plant_scale, mode="bilinear", align_corners=False)
    g = seg(model, torch.cat([xs, xs.flip(-1)]))
    out = {"g": norm(up((g[0] + g[1].flip(-1)) / 2, SIZE))}
    if tiles:
        h = cfg.sem_tile
        acc, cnt = torch.zeros((3, SIZE, SIZE), device=x.device), torch.zeros((1, SIZE, SIZE), device=x.device)
        tl = [(t, l) for t in tile_starts(h) for l in tile_starts(h)]
        for a in range(0, len(tl), cfg.tile_batch):
            batch = tl[a:a + cfg.tile_batch]
            p = seg(model, torch.cat([F.interpolate(x[:, :, t:t + h, l:l + h], size=(2 * h, 2 * h), mode="bilinear",
                                                    align_corners=False) for t, l in batch]))
            for (t, l), pj in zip(batch, p):
                acc[:, t:t + h, l:l + h] += up(pj / pj.sum(0, keepdim=True).clamp(min=1e-6), h)
                cnt[:, t:t + h, l:l + h] += 1
        out["t"] = norm(acc / cnt)
    return out


@torch.no_grad()
def quick_iou(cfg, model, cache, idx, dev):
    model.eval()
    conf = np.zeros((3, 3), np.int64)
    for i in idx:
        p = views(cfg, model, image_tensor(cache["img_val"][i], dev), tiles=False)["g"]
        conf += confusion(np.asarray(cache["sem_val"][i]), p.argmax(0).cpu().numpy())
    return float(ious(conf)[2])
