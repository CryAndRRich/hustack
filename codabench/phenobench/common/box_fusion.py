import numpy as np
import torch
from torchvision.ops import box_iou

from common.box_metric import SIZE, outside_partials


def to_xyxy(b):
    b = np.asarray(b, np.float64).reshape(-1, 4)
    return np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], 1)


def to_cxcywh(x):
    x = np.clip(np.asarray(x, np.float64).reshape(-1, 4), -0.5, SIZE - 0.5)
    cx = np.clip(np.round((x[:, 0] + x[:, 2]) / 2), 0, SIZE - 1)
    cy = np.clip(np.round((x[:, 1] + x[:, 3]) / 2), 0, SIZE - 1)
    w = np.minimum(np.round(x[:, 2] - x[:, 0]), np.minimum(2 * cx, 2 * (SIZE - 1 - cx)) + 1)
    h = np.minimum(np.round(x[:, 3] - x[:, 1]), np.minimum(2 * cy, 2 * (SIZE - 1 - cy)) + 1)
    return np.stack([cx, cy, w, h], 1).astype(np.int64)


def _iou(b, fb):
    iw = np.clip(np.minimum(b[2], fb[:, 2]) - np.maximum(b[0], fb[:, 0]), 0, None)
    ih = np.clip(np.minimum(b[3], fb[:, 3]) - np.maximum(b[1], fb[:, 1]), 0, None)
    inter = iw * ih
    return inter / np.maximum((b[2] - b[0]) * (b[3] - b[1]) + (fb[:, 2] - fb[:, 0]) * (fb[:, 3] - fb[:, 1]) - inter, 1e-9)


def wbf(sources, weights, thr):
    w = np.asarray(weights, np.float64)
    lab = np.concatenate([np.asarray(s[0], np.int64) for s in sources])
    box = np.concatenate([np.asarray(s[1], np.float64).reshape(-1, 4) for s in sources])
    sc = np.concatenate([np.asarray(s[2], np.float64) for s in sources])
    src = np.concatenate([np.full(len(s[2]), k) for k, s in enumerate(sources)]).astype(np.int64)
    ref = np.concatenate([np.arange(len(s[2])) for s in sources]).astype(np.int64)
    out = [[], [], [], []]
    for c in np.unique(lab):
        idx = np.flatnonzero(lab == c)
        idx = idx[np.argsort(-sc[idx] * w[src[idx]], kind="stable")]
        fb, acc, wsum, best, mem = np.zeros((0, 4)), [], [], [], []
        for j in idx:
            k = -1
            if len(fb):
                iou = _iou(box[j], fb)
                k = int(iou.argmax()) if iou.max() > thr else -1
            if k < 0:
                fb = np.vstack([fb, box[j]])
                acc.append(np.zeros(4))
                wsum.append(0.0)
                best.append(np.zeros(len(w)))
                mem.append(np.full(len(w), -1))
                k = len(fb) - 1
            wt = max(sc[j] * w[src[j]], 1e-6)
            acc[k] = acc[k] + wt * box[j]
            wsum[k] += wt
            fb[k] = acc[k] / wsum[k]
            if mem[k][src[j]] < 0 or sc[j] > best[k][src[j]]:
                best[k][src[j]], mem[k][src[j]] = sc[j], ref[j]
        out[0].append(np.full(len(fb), c))
        out[1].append(fb)
        out[2].append(np.array(best).reshape(-1, len(w)))
        out[3].append(np.array(mem, np.int64).reshape(-1, len(w)))
    if not out[0]:
        return np.zeros(0, np.int64), np.zeros((0, 4)), np.zeros(0), np.zeros((0, len(w))), np.zeros((0, len(w)), np.int64)
    labels, fb, best, mem = (np.concatenate(o) for o in out)
    return labels, fb, (best * w).sum(1) / w.sum(), best, mem


def integral(lab):
    return [np.pad((lab == k).cumsum(0).cumsum(1), ((1, 0), (1, 0))) for k in range(3)]


def features(fused, extras, ii):
    labels, xyxy, score, best, mem = fused
    b = to_cxcywh(xyxy)
    x = np.clip(np.round(xyxy), 0, SIZE - 1).astype(np.int64)
    area = (x[:, 2] - x[:, 0] + 1) * (x[:, 3] - x[:, 1] + 1)
    frac = [(i[x[:, 3] + 1, x[:, 2] + 1] - i[x[:, 1], x[:, 2] + 1] - i[x[:, 3] + 1, x[:, 0]] + i[x[:, 1], x[:, 0]]) / area
            for i in ii]
    dup = np.zeros(len(labels))
    for c in np.unique(labels):
        sel = np.flatnonzero(labels == c)
        if len(sel) > 1:
            iou = box_iou(torch.from_numpy(xyxy[sel]), torch.from_numpy(xyxy[sel])).numpy()
            higher = score[sel][None, :] > score[sel][:, None]
            dup[sel] = np.where(higher, iou, 0).max(1)
    m = mem[:, 0]
    ex = [np.where(m >= 0, v[np.maximum(m, 0)], -1.0) if len(v) else np.full(len(m), -1.0) for v in extras.values()]
    cols = [labels, score, *best.T, np.log1p(b[:, 2] * b[:, 3]), (b[:, 2] + 1) / (b[:, 3] + 1), *frac, dup, *ex]
    return np.stack(cols, 1).astype(np.float64)


def targets(boxes, labels, gt):
    t = np.zeros(len(labels))
    for c in np.unique(labels):
        sel, g = np.flatnonzero(labels == c), gt["boxes"][gt["labels"] == c]
        if len(g):
            t[sel] = box_iou(torch.from_numpy(to_xyxy(boxes[sel])), torch.from_numpy(to_xyxy(g))).numpy().max(1)
    return t, outside_partials(boxes, gt["partial"])


def fit(X, y):
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=50,
                                         l2_regularization=1.0, random_state=0).fit(X, y)


def rescored(model, X):
    return np.clip(model.predict(X), 1e-4, 1.0) + 1e-6 * X[:, 1]
