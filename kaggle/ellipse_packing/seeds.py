import math

import numpy as np

from geometry import N_ELLIPSES

SQRT3 = math.sqrt(3.0)

ASPECT_FAN = (0.8, 2.0)

ANGLE_NOISE_FAN = (math.radians(5.0), math.radians(35.0))

LOCAL_DOMAIN_BLOCK = 3


def row_structures(n=N_ELLIPSES):
    out = []
    for w in range(2, n + 1):
        for n_long in range(0, n + 1):
            rem = n - n_long * w
            if rem < 0 or w == 1:
                continue
            if rem % (w - 1) != 0:
                continue
            n_short = rem // (w - 1)
            rows = n_long + n_short
            if rows == 0:
                continue
            lo, hi = min(n_long, n_short), max(n_long, n_short)
            if rows == 1:
                gaps = 0.0
            else:
                alternating = 2 * lo if hi > lo else rows - 1
                same = (rows - 1) - alternating
                if same < 0:
                    continue
                gaps = alternating * SQRT3 + same * 2.0
            out.append((2.0 * w * (2.0 + gaps), w, n_long, n_short))
    out.sort()
    return out


def hex_layout(w, n_long, n_short):
    labels = []
    a, b = ("L", "S") if n_long >= n_short else ("S", "L")
    na, nb = max(n_long, n_short), min(n_long, n_short)
    for k in range(na):
        labels.append(a)
        if k < nb:
            labels.append(b)
    centres, y = [], 1.0
    for k, lab in enumerate(labels):
        if k:
            y += SQRT3 if labels[k - 1] != lab else 2.0
        count = w if lab == "L" else w - 1
        x0 = 1.0 if lab == "L" else 2.0
        centres += [(x0 + 2.0 * c, y) for c in range(count)]
    return np.array(centres, dtype=np.float64), 2.0 * w, y + 1.0


def hex67():
    uv, Wu, Hv = hex_layout(10, 4, 3)
    assert len(uv) == N_ELLIPSES, len(uv)
    return uv, Wu, Hv


def _inflate(uv, Wu, Hv, factor):
    return uv * factor, Wu * factor, Hv * factor


def _row_ids(uv):
    order = np.argsort(uv[:, 1])
    gaps = np.diff(uv[order, 1])
    breaks = gaps > 3.0 * np.median(gaps)
    row_of_sorted = np.concatenate([[0], np.cumsum(breaks)])
    row = np.empty(len(uv), dtype=np.int64)
    row[order] = row_of_sorted
    return row


def _point_pairs(uv):
    centroid = uv.mean(axis=0)
    mirrored = 2.0 * centroid - uv
    d2 = ((uv[:, None, :] - mirrored[None, :, :]) ** 2).sum(-1)
    return d2.argmin(axis=0)


def _patch_ids(uv, block=LOCAL_DOMAIN_BLOCK):
    row = _row_ids(uv)
    patch = np.empty(len(uv), dtype=np.int64)
    next_id = 0
    for r in np.unique(row):
        idx = np.where(row == r)[0]
        idx = idx[np.argsort(uv[idx, 0])]
        for start in range(0, len(idx), block):
            patch[idx[start:start + block]] = next_id
            next_id += 1
    return patch


def _draw_loguniform(rng, bounds):
    lo, hi = bounds
    return math.exp(rng.uniform(math.log(lo), math.log(hi)))


def retarget_aspect(uv, Wu, Hv, aspect):
    sx = math.sqrt(aspect / (Wu / Hv))
    sy = 1.0 / sx
    grow = 1.0 / min(sx, sy)
    sx, sy = sx * grow, sy * grow
    return uv * np.array([sx, sy]), Wu * sx, Hv * sy


def make_population(n, rng, slack=1.002, fan=ASPECT_FAN, angle_fan=ANGLE_NOISE_FAN):
    hex_uv, hex_W, hex_H = hex67()
    alts = [s for s in row_structures() if s[1] >= 4][:6]

    uv = np.zeros((n, N_ELLIPSES, 2))
    th = np.zeros((n, N_ELLIPSES))
    Wu = np.zeros(n)
    Hv = np.zeros(n)

    row = _row_ids(hex_uv)
    partner = _point_pairs(hex_uv)
    patch = _patch_ids(hex_uv)
    for i in range(n):
        kind = 0 if i == 0 else 1 + (i - 1) % 7
        if kind == 0:
            u, W, H = _inflate(hex_uv, hex_W, hex_H, slack)
            t = np.zeros(N_ELLIPSES)
        elif kind == 1:
            u, W, H = _inflate(hex_uv, hex_W, hex_H, 1.03)
            u, W, H = retarget_aspect(u, W, H, _draw_loguniform(rng, fan))
            u = u + rng.normal(scale=0.03, size=u.shape)
            t = rng.normal(scale=_draw_loguniform(rng, angle_fan), size=N_ELLIPSES)
        elif kind == 2:
            u, W, H = _inflate(hex_uv, hex_W, hex_H, 1.12)
            u, W, H = retarget_aspect(u, W, H, _draw_loguniform(rng, fan))
            t = np.zeros(N_ELLIPSES)
            flip = rng.random(N_ELLIPSES) < rng.uniform(0.08, 0.35)
            t[flip] = np.pi / 2
            t += rng.normal(scale=0.05, size=N_ELLIPSES)
        elif kind == 3:
            _, w, nl, ns = alts[rng.integers(len(alts))]
            u, W, H = hex_layout(w, nl, ns)
            u, W, H = _inflate(u, W, H, 1.05)
            u, W, H = retarget_aspect(u, W, H, _draw_loguniform(rng, fan))
            u = u + rng.normal(scale=0.03, size=u.shape)
            t = rng.normal(scale=_draw_loguniform(rng, angle_fan), size=N_ELLIPSES)
        elif kind == 4:
            u, W, H = _inflate(hex_uv, hex_W, hex_H, 1.08)
            u, W, H = retarget_aspect(u, W, H, _draw_loguniform(rng, fan))
            t = np.where(row % 2 == 0, 0.0, np.pi / 2)
            t = t + rng.normal(scale=_draw_loguniform(rng, angle_fan), size=N_ELLIPSES)
        elif kind == 5:
            u, W, H = _inflate(hex_uv, hex_W, hex_H, 1.05)
            u, W, H = retarget_aspect(u, W, H, _draw_loguniform(rng, fan))
            raw = rng.normal(scale=_draw_loguniform(rng, angle_fan), size=N_ELLIPSES)
            canon = np.minimum(np.arange(N_ELLIPSES), partner)
            t = raw[canon]
        elif kind == 6:
            u, W, H = _inflate(hex_uv, hex_W, hex_H, 1.05)
            u, W, H = retarget_aspect(u, W, H, _draw_loguniform(rng, fan))
            patch_draw = rng.normal(scale=_draw_loguniform(rng, angle_fan), size=patch.max() + 1)
            t = patch_draw[patch]
        else:
            area = N_ELLIPSES * math.pi / rng.uniform(0.40, 0.55)
            aspect = _draw_loguniform(rng, fan)
            W = math.sqrt(area * aspect)
            H = area / W
            u = np.stack([rng.uniform(1.2, W - 1.2, N_ELLIPSES),
                          rng.uniform(1.2, H - 1.2, N_ELLIPSES)], axis=1)
            t = rng.uniform(0.0, np.pi, N_ELLIPSES)
        uv[i], th[i], Wu[i], Hv[i] = u, t, W, H
    return uv, th, Wu, Hv
