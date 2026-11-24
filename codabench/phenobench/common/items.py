import pickle

import numpy as np
from scipy import ndimage

SIZE = 1024


def area(it):
    return int(it[2].sum())


def box(it):
    return it[0], it[1], it[0] + it[2].shape[0], it[1] + it[2].shape[1]


def from_probs(prob, score, min_score, y_off=0, x_off=0):
    keep = (score >= min_score).nonzero()[:, 0]
    if keep.numel() == 0:
        return []
    binm = (prob[keep] > 0.5).cpu().numpy()
    sc = score[keep].float().cpu().numpy()
    rows, cols = binm.any(2), binm.any(1)
    out = []
    for k in range(len(sc)):
        r, c = np.flatnonzero(rows[k]), np.flatnonzero(cols[k])
        if len(r):
            out.append((y_off + int(r[0]), x_off + int(c[0]), binm[k, r[0]:r[-1] + 1, c[0]:c[-1] + 1], float(sc[k]), None))
    return out


def by_score(items):
    return sorted(items, key=lambda it: -it[3])


def paint(items, keep_frac, min_score, size=SIZE):
    canvas = np.zeros((size, size), np.int32)
    kept, kept_items = [], []
    for it in items:
        y0, x0, m, s = it[:4]
        if s < min_score:
            break
        reg = canvas[y0:y0 + m.shape[0], x0:x0 + m.shape[1]]
        free = m & (reg == 0)
        fa = int(free.sum())
        if fa == 0 or fa < keep_frac * m.sum():
            continue
        kept.append(s)
        kept_items.append(it)
        reg[free] = len(kept)
    return canvas, np.array(kept), kept_items


def intersection(a, b):
    ay0, ax0, ay1, ax1 = box(a)
    by0, bx0, by1, bx1 = box(b)
    y0, x0, y1, x1 = max(ay0, by0), max(ax0, bx0), min(ay1, by1), min(ax1, bx1)
    if y0 >= y1 or x0 >= x1:
        return 0
    return int((a[2][y0 - ay0:y1 - ay0, x0 - ax0:x1 - ax0] & b[2][y0 - by0:y1 - by0, x0 - bx0:x1 - bx0]).sum())


def iou(a, b):
    inter = intersection(a, b)
    return inter / max(int(a[2].sum()) + int(b[2].sum()) - inter, 1)


def boxes_array(items):
    return np.array([box(it) for it in items], dtype=np.int64).reshape(-1, 4)


def overlapping(bxs, b):
    return np.flatnonzero((bxs[:, 0] < b[2]) & (bxs[:, 2] > b[0]) & (bxs[:, 1] < b[3]) & (bxs[:, 3] > b[1]))


def merge_zoom(glob, zoom, gate, contain=False):
    small_z = [z for z in zoom if area(z) < gate]
    if contain:
        big = [g for g in glob if area(g) >= gate]
        bb = boxes_array(big)
        small_z = [z for z in small_z if not any(intersection(z, big[j]) >= 0.5 * area(z) for j in overlapping(bb, box(z)))]
    zb = boxes_array(small_z)
    out = list(small_z)
    for g in glob:
        if area(g) >= gate or not any(iou(g, small_z[j]) > 0.5 for j in overlapping(zb, box(g))):
            out.append(g)
    return by_score(out)


def size_keep(canvas, scores, thr, offset, split):
    n = len(scores)
    small = np.bincount(canvas.ravel(), minlength=n + 1)[1:n + 1] < split
    return np.asarray(scores) >= np.where(small, thr + offset, thr)


def largest_component(m):
    lab, n = ndimage.label(m)
    if n <= 1:
        return m
    return lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)


def place(mask, top, left, size=SIZE):
    ys, xs = max(top, 0), max(left, 0)
    ye, xe = min(top + mask.shape[0], size), min(left + mask.shape[1], size)
    if ys >= ye or xs >= xe:
        return None
    m = mask[ys - top:ye - top, xs - left:xe - left]
    rows, cols = np.flatnonzero(m.any(1)), np.flatnonzero(m.any(0))
    if len(rows) == 0:
        return None
    return ys + int(rows[0]), xs + int(cols[0]), m[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]


def save(items, path):
    rows = [(y0, x0, m.shape, np.packbits(m), s, tag) for y0, x0, m, s, tag in items]
    path.write_bytes(pickle.dumps(rows, protocol=4))


def load(path):
    return [(y0, x0, np.unpackbits(p, count=shape[0] * shape[1]).reshape(shape).astype(bool), s, tag)
            for y0, x0, shape, p, s, tag in pickle.loads(path.read_bytes())]
