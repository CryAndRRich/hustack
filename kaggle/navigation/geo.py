import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.special import expit

from preprocess import epoch_index, pieces

G, RC, SIG_PRIOR = 2.0, 25, 15.0
N1 = 2 * RC + 1
OFF = np.array([(i, j) for i in range(-RC, RC + 1) for j in range(-RC, RC + 1)])
CENTER = len(OFF) // 2
RING8 = [(4, 0), (-4, 0), (0, 4), (0, -4), (3, 3), (3, -3), (-3, 3), (-3, -3)]
RING16 = [(8, 0), (-8, 0), (0, 8), (0, -8), (6, 6), (6, -6), (-6, 6), (-6, -6)]
SHIFT = 2 ** 20


def cell(i, j):
    return (i + RC) * N1 + j + RC


def visibility(m, vis):
    return expit(vis[0] * np.clip(m, -30, 30) / 10 + vis[1])


class Cache:
    def __init__(self, city, h_ant):
        self.city, self.h_ant = city, h_ant
        self.keys, self.inside = np.zeros(0, np.int64), np.zeros(0, bool)
        self.msk = np.zeros((0, 360), np.float32)

    def rows(self, keys):
        new = np.setdiff1d(np.unique(keys), self.keys)
        if len(new):
            cx, cy = (new // 2 ** 21 - SHIFT) * G, (new % 2 ** 21 - SHIFT) * G
            allk = np.r_[self.keys, new]
            order = np.argsort(allk)
            self.msk = np.concatenate([self.msk, self.city.masks_at(cx, cy, self.h_ant)])[order]
            self.inside = np.r_[self.inside, self.city.inside(cx, cy)][order]
            self.keys = allk[order]
        return np.searchsorted(self.keys, keys)


class Lattice:
    def __init__(self, df, ep, cache, cx, cy, vis):
        self.cache = cache
        self.ok = np.isfinite(cx) & np.isfinite(cy)
        self.bx = np.where(self.ok, np.round(cx / G), 0).astype(np.int64)
        self.by = np.where(self.ok, np.round(cy / G), 0).astype(np.int64)
        keys = (self.bx[:, None] + OFF[:, 0] + SHIFT) * 2 ** 21 + (self.by[:, None] + OFF[:, 1] + SHIFT)
        self.rows = cache.rows(keys)
        self.inside = cache.inside[self.rows]
        self.b = np.mod(np.floor(df.az_h.values).astype(int), 360)
        self.el = np.radians(df.el_h.values)
        ek = epoch_index(df, ep[["seg", "t"]])
        groups = {int(k): v for k, v in pd.Series(np.arange(len(df))).groupby(ek).indices.items()}
        self.by_epoch = {k: v for k, v in groups.items() if self.ok[k]}
        self.los, self.pb = {}, {}
        for k, r in self.by_epoch.items():
            m = self.margin(k, r)
            self.los[k], self.pb[k] = m > 0, visibility(m, vis).astype(np.float32)
        self.pid = pieces(ep)
        self.t = ep.t.values.astype(float)
        self.v = np.stack([ep.ve.values, ep.vn.values], 1)

    def margin(self, k, r, cells=None):
        rows = self.rows[k] if cells is None else self.rows[k][cells]
        return np.degrees(self.el[r][None, :] - self.cache.msk[rows[:, None], self.b[r][None, :]])

    def sm_ll(self, p):
        pc = np.clip(p, 0.02, 0.98)
        return {k: np.log(pc[r] * self.pb[k] + (1 - pc[r]) * (1 - self.pb[k])).sum(1)
                for k, r in self.by_epoch.items()}

    def smooth(self, ll, w, eps):
        gamma = {}
        prior = np.exp(-((OFF * G) ** 2).sum(1) / (2 * SIG_PRIOR ** 2)).reshape(N1, N1)
        for q in np.unique(self.pid):
            ks = [k for k in np.flatnonzero(self.pid == q) if k in ll]
            if not ks:
                continue
            like = []
            for k in ks:
                z = np.where(self.inside[k], -np.inf, w * ll[k])
                z = np.exp(z - z.max()) if np.isfinite(z.max()) else np.ones(len(OFF))
                like.append(z.reshape(N1, N1))
            steps = [None]
            for a, b in zip(ks[:-1], ks[1:]):
                dt = self.t[b] - self.t[a]
                v = 0.5 * (self.v[a] + self.v[b])
                known = np.isfinite(v).all()
                s = np.array([self.bx[a] - self.bx[b], self.by[a] - self.by[b]], float)
                s += v * dt / G if known else 0.0
                steps.append((s, np.hypot(0.5, (0.3 if known else 3.0) * dt) / G))
            alpha = [norm(prior * like[0])]
            for i in range(1, len(ks)):
                s, sg = steps[i]
                pr = norm(ndimage.gaussian_filter(ndimage.shift(alpha[-1], s, order=1, mode="constant"), sg,
                                                  mode="constant"))
                alpha.append(norm(((1 - eps) * pr + eps / pr.size) * like[i]))
            beta = np.ones((N1, N1))
            for i in range(len(ks) - 1, -1, -1):
                gamma[ks[i]] = norm(alpha[i] * beta).ravel()
                if i > 0:
                    s, sg = steps[i]
                    wb = like[i] * beta
                    beta = norm((1 - eps) * ndimage.shift(ndimage.gaussian_filter(wb, sg, mode="constant"), -s,
                                                          order=1, mode="constant") + eps * wb.mean())
        return gamma

    def mean_position(self, gamma):
        x, y = np.full(len(self.ok), np.nan), np.full(len(self.ok), np.nan)
        for k, g in gamma.items():
            m = g @ OFF
            x[k], y[k] = (self.bx[k] + m[0]) * G, (self.by[k] + m[1]) * G
        return x, y

    def base_features(self, df):
        ring8, ring16 = [cell(i, j) for i, j in RING8], [cell(i, j) for i, j in RING16]
        near = np.array([CENTER] + ring8 + ring16)
        f = {c: np.full(len(df), np.nan) for c in ("m_c", "los8", "los16", "m_mean", "m_max", "m_min")}
        for k, r in self.by_epoch.items():
            marg = self.margin(k, r, near)
            los = marg > 0
            f["m_c"][r] = marg[0]
            f["los8"][r], f["los16"][r] = los[1:9].mean(0), los[9:].mean(0)
            f["m_mean"][r], f["m_max"][r], f["m_min"][r] = marg.mean(0), marg.max(0), marg.min(0)
        for c, v in f.items():
            df[c] = v

    def post_features(self, df, ep, gamma, sfx):
        names = ("post_los", "post_hard", "pm_m", "map_m", "post_std", "post_conf", "pm_shift")
        f = {c: np.full(len(df), np.nan) for c in names}
        x, y = self.mean_position(gamma)
        shift = np.hypot(x - ep.x.values, y - ep.y.values)
        for k, g in gamma.items():
            r = self.by_epoch[k]
            m = g @ OFF
            pm = cell(*np.clip(np.round(m).astype(int), -RC, RC))
            marg = self.margin(k, r, np.array([pm, int(np.argmax(g))]))
            f["post_los"][r], f["post_hard"][r] = g @ self.pb[k], g @ self.los[k]
            f["pm_m"][r], f["map_m"][r] = marg[0], marg[1]
            f["post_std"][r] = np.sqrt(g @ (((OFF - m) * G) ** 2).sum(1))
            f["post_conf"][r] = g.max()
            f["pm_shift"][r] = shift[k]
        for c, v in f.items():
            df[c + sfx] = v
        ep["x" + sfx], ep["y" + sfx] = x, y
        return [c + sfx for c in names]


def norm(a):
    s = a.sum()
    return a / s if s > 0 and np.isfinite(s) else np.full_like(a, 1.0 / a.size)
