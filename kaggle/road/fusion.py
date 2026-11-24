from itertools import combinations

import numpy as np

FLIP = "~f"


def arms(models, max_subset):
    sizes = sorted({*range(1, min(max_subset, len(models)) + 1), len(models)})
    out = {}
    for k in sizes:
        for subset in combinations(models, k):
            for flip in (False, True):
                members = [m + suffix for m in subset for suffix in (("", FLIP) if flip else ("",))]
                out["+".join(subset) + ("_flip" if flip else "")] = {"models": list(subset), "flip": flip,
                                                                     "members": members}
    return out


def pairwise_iou(a, b):
    iw = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = iw * ih
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter)


def fuse_class(d, member, n_parts, iou):
    overlap = pairwise_iou(d[:, 2:], d[:, 2:])
    suppressed = np.zeros(len(d), bool)
    seeds = []
    for i in range(len(d)):
        if not suppressed[i]:
            seeds.append(i)
            suppressed |= overlap[i] > iou
    owner = overlap[seeds].argmax(0)
    n = len(seeds)
    weight = np.bincount(owner, weights=d[:, 1], minlength=n)
    coords = np.stack([np.bincount(owner, weights=d[:, 1] * d[:, 2 + k], minlength=n) for k in range(4)], 1)
    count = np.bincount(owner, minlength=n)
    members_in = np.bincount(np.unique(owner * n_parts + member) // n_parts, minlength=n)
    return np.column_stack([d[seeds, 0], weight / count * members_in / n_parts, coords / weight[:, None]])


def fuse(parts, iou, max_det):
    if len(parts) == 1:
        return parts[0]
    d = np.concatenate(parts)
    if not len(d):
        return np.zeros((0, 6))
    member = np.repeat(np.arange(len(parts)), [len(p) for p in parts])
    order = np.argsort(-d[:, 1], kind="stable")
    d, member = d[order], member[order]
    out = np.concatenate([fuse_class(d[d[:, 0] == c], member[d[:, 0] == c], len(parts), iou)
                          for c in np.unique(d[:, 0])])
    out = out[np.argsort(-out[:, 1], kind="stable")[:max_det]]
    return out[(out[:, 4] - out[:, 2] >= 0.05) & (out[:, 5] - out[:, 3] >= 0.05)]
