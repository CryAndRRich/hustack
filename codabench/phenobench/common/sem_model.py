import dataclasses
import math
import random

import numpy as np
import torch
import torch.nn.functional as F

from common.data import SIZE
from common.models import loss_in_fp32
from common.plant import image_tensor, tile_starts
from common.sem_metric import confusion, ious
from common.semantic import LABELS, SemSet, norm, seg, up


def load_sem(kind, path, size):
    from transformers import EomtDinov3ForUniversalSegmentation, Mask2FormerForUniversalSegmentation
    kw = {"id2label": LABELS, "label2id": {v: k for k, v in LABELS.items()}, "ignore_mismatched_sizes": True,
          "dtype": torch.float32}
    if kind == "m2f":
        model = Mask2FormerForUniversalSegmentation.from_pretrained(path, **kw)
    else:
        model = EomtDinov3ForUniversalSegmentation.from_pretrained(path, image_size=size, **kw)
        bufs = [b for n, b in model.named_buffers() if n.endswith("attn_mask_probs")]
        assert len(bufs) == 1, f"expected one attn_mask_probs buffer, found {len(bufs)}"
        bufs[0].zero_()
    assert model.config.num_labels == 3, model.config.num_labels
    ew = [b for n, b in model.named_buffers() if n.endswith("empty_weight")]
    assert len(ew) == 1, f"expected one empty_weight buffer, found {len(ew)}"
    with torch.no_grad():
        ew[0].copy_(torch.tensor([1.0, 1.0, 1.0, model.config.no_object_weight]))
    model.loss_fp32 = loss_in_fp32(model)
    return model


class SemData(SemSet):
    def __init__(self, cfg, cache_dir, spec):
        super().__init__(dataclasses.replace(cfg, m2f_size=spec["size"], m2f_stride=spec["stride"],
                                             sem_zoom_min=spec["zoom"][0], sem_zoom_max=spec["zoom"][1]), cache_dir)
        self.gamma = cfg.sem_gamma

    def __getitem__(self, n):
        img, lab = super().__getitem__(n)
        g = math.exp(random.uniform(-math.log(self.gamma), math.log(self.gamma)))
        return (255 * (img.float() / 255) ** g).round().to(torch.uint8), lab


@torch.no_grad()
def view(model, x, h, size, batch):
    if h == SIZE:
        xs = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
        p = seg(model, torch.cat([xs, xs.flip(-1)]))
        return norm(up((p[0] + p[1].flip(-1)) / 2, SIZE))
    acc, cnt = torch.zeros((3, SIZE, SIZE), device=x.device), torch.zeros((1, SIZE, SIZE), device=x.device)
    tl = [(t, l) for t in tile_starts(h) for l in tile_starts(h)]
    for a in range(0, len(tl), batch):
        part = tl[a:a + batch]
        p = seg(model, torch.cat([F.interpolate(x[:, :, t:t + h, l:l + h], size=(size, size), mode="bilinear",
                                                align_corners=False) for t, l in part]))
        for (t, l), pj in zip(part, p):
            acc[:, t:t + h, l:l + h] += up(pj / pj.sum(0, keepdim=True).clamp(min=1e-6), h)
            cnt[:, t:t + h, l:l + h] += 1
    return norm(acc / cnt)


@torch.no_grad()
def quick_iou(model, cache, idx, h, size, batch, dev):
    model.eval()
    conf = np.zeros((3, 3), np.int64)
    for i in idx:
        p = view(model, image_tensor(cache["img_val"][i], dev), h, size, batch)
        conf += confusion(np.asarray(cache["sem_val"][i]), p.argmax(0).cpu().numpy())
    return float(ious(conf)[2])
