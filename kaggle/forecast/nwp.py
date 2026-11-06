import warnings
from itertools import combinations

import numpy as np
import pandas as pd

from preprocess import CTX, H, VARS, season

NWP_MODELS = ["ecmwf_ifs025", "ecmwf_open", "ecmwf_hres", "icon_global", "gem_global", "gefs", "gfs_global", "jma_gsm"]
AR_DAYS = 5
LAD_ITERS = 20
MIN_PAIRS = 10
CV_FOLDS = 5
SEASON_SHRINK = 200


def load_nwp(paths, dates, S):
    df = pd.concat([pd.read_csv(p, parse_dates=["date"]) for p in paths])
    t0 = (df.date.min() - dates[0]).days
    N = np.full((len(NWP_MODELS), S, len(dates) - t0, H, len(VARS)), np.nan, np.float32)
    day = (df.date - dates[0]).dt.days.to_numpy() - t0
    m = df.model.map({k: i for i, k in enumerate(NWP_MODELS)}).to_numpy()
    s = df.station.str[1:].astype(int).to_numpy()
    N[m, s, day, df.lead.to_numpy() - 1] = df[VARS].to_numpy()
    return N, t0


def fit_ar(Z, n_fit):
    B = []
    t = np.arange(AR_DAYS - 1, n_fit - H)
    for s in range(Z.shape[0]):
        F = np.column_stack([np.ones(len(t))] + [Z[s, t - j] for j in range(AR_DAYS)])
        Y = np.concatenate([Z[s, t + h] for h in range(1, H + 1)], 1)
        B.append(np.linalg.lstsq(F, Y, rcond=None)[0])
    return np.array(B)


def ar_predict(B, Z, s, i):
    F = np.column_stack([np.ones(len(s))] + [Z[s, i - j] for j in range(AR_DAYS)])
    return np.einsum("np,npq->nq", F, B[s]).reshape(len(s), H, len(VARS))


def lad(F, y):
    b = np.linalg.lstsq(F, y, rcond=None)[0]
    for _ in range(LAD_ITERS):
        w = 1 / np.maximum(np.abs(y - F @ b), 1e-3)
        b = np.linalg.lstsq(F * w[:, None], y * w, rcond=None)[0]
    return b


class Stack:
    def __init__(self, fr, N, t0, windows, n_train, dates):
        self.fr, self.N, self.t0 = fr, N, t0
        self.seas = season(dates)[t0:]
        self.err = N - fr.X[None, :, t0:, None, :]
        self.fold = (np.arange(t0, len(dates)) // 45) % CV_FOLDS
        B = fit_ar(fr.Z, n_train)
        rows = []
        for s, e in zip(windows.station, windows.end):
            for i in range(e - CTX + AR_DAYS, e):
                for L in range(1, e - i + 1):
                    rows.append((s, i, L))
        n_ctx = len(rows)
        runs = t0 - 1 + np.flatnonzero((~np.isnan(N[:, :, :n_train - t0, 0, 0])).any((0, 1)))
        rows += [(s, i, L) for i in runs for s in range(N.shape[1]) for L in range(1, H + 1)]
        s, i, L = map(np.array, zip(*rows))
        self.ctx = np.arange(len(s)) < n_ctx
        self.pairs = (s, i, L, ar_predict(B, fr.Z, s, i)[np.arange(len(s)), L - 1])
        self.bias = self._bias(np.ones(len(self.seas), bool))
        self.coef = self._fit_all(np.ones(len(s), bool))

    def _bias(self, days):
        err = np.where(days[None, None, :, None, None], self.err, np.nan)
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            lead = np.nan_to_num(np.nanmedian(err, 2))
            seas = np.zeros(err.shape[:2] + (4, err.shape[-1]))
            if SEASON_SHRINK is not None:
                for k in range(4):
                    e = (err[:, :, self.seas == k] - lead[:, :, None]).reshape(err.shape[0], err.shape[1], -1, err.shape[-1])
                    cnt = (~np.isnan(e)).sum(2)
                    seas[:, :, k] = np.nan_to_num(np.nanmedian(e, 2)) * cnt / (cnt + SEASON_SHRINK)
        return lead, seas

    def anomalies(self, s, c, lead, v, bias=None):
        b_lead, b_seas = self.bias if bias is None else bias
        return (self.N[:, s, c - self.t0, lead - 1, v].T - self.fr.base[s, c, v][:, None] - b_lead[:, s, lead - 1, v].T
                - b_seas[:, s, self.seas[c - self.t0], v].T)

    def _fit_all(self, use, bias=None):
        s, i, L, P = self.pairs
        coef = {}
        for v in range(len(VARS)):
            for lead in range(1, H + 1):
                k = use & (L == lead)
                c = i[k] + lead
                y, p = self.fr.Z[s[k], c, v], P[k, v]
                n = self.anomalies(s[k], c, lead, v, bias)
                have = ~np.isnan(n)
                keys = {sub for row in {tuple(np.flatnonzero(r)) for r in have}
                        for size in range(1, len(row) + 1) for sub in combinations(row, size)}
                for key in keys:
                    rows = have[:, list(key)].all(1)
                    if rows.sum() >= MIN_PAIRS * (1 + len(key)):
                        coef[(v, lead, key)] = lad(np.column_stack([p[rows], n[rows][:, list(key)]]), y[rows])
        return coef

    def _apply(self, coef, v, lead, n, g):
        out = g.copy()
        fitted = [key for (vv, ll, key) in coef if vv == v and ll == lead]
        for j, row in enumerate(~np.isnan(n)):
            avail = set(np.flatnonzero(row))
            best = max((k for k in fitted if avail.issuperset(k)), key=lambda k: (len(k), [-m for m in k]), default=None)
            if best is not None:
                b = coef[(v, lead, best)]
                out[j] = b[0] * g[j] + n[j, list(best)] @ b[1:]
        return out

    def combine(self, s, e, G):
        out = G.copy()
        for v in range(len(VARS)):
            for lead in range(1, H + 1):
                out[:, lead - 1, v] = self._apply(self.coef, v, lead, self.anomalies(s, e + lead, lead, v), G[:, lead - 1, v])
        return out

    def cv_report(self, scales):
        s, i, L, P = self.pairs
        fold = self.fold[i - self.t0]
        stats, stacked = np.zeros((len(s), len(VARS))), np.zeros((len(s), len(VARS)))
        for f in range(CV_FOLDS):
            bias = self._bias(self.fold != f)
            coef = self._fit_all(fold != f, bias)
            for v in range(len(VARS)):
                for lead in range(1, H + 1):
                    k = np.flatnonzero((fold == f) & (L == lead))
                    c = i[k] + lead
                    y = self.fr.Z[s[k], c, v]
                    stats[k, v] = np.abs(P[k, v] - y) / scales[v]
                    pred = self._apply(coef, v, lead, self.anomalies(s[k], c, lead, v, bias), P[k, v])
                    stacked[k, v] = np.abs(pred - y) / scales[v]
        has = ~np.isnan(self.N[:, s, i + L - self.t0, L - 1, 0]).T
        groups = [("all pairs", self.ctx)]
        groups += [(f"best: {m}", self.ctx & has[:, j] & ~has[:, :j].any(1)) for j, m in enumerate(NWP_MODELS)]
        groups += [("no NWP", self.ctx & ~has.any(1)), ("training runs", ~self.ctx)]
        for name, k in groups:
            if k.any():
                print(f"  {name:22s} n={k.sum():5d}  proxy {stats[k].mean():.4f}  stacked {stacked[k].mean():.4f}  "
                      f"{dict(zip(VARS, stacked[k].mean(0).round(3)))}")
        for lead in range(1, H + 1):
            k = self.ctx & (L == lead)
            print(f"  lead {lead}: proxy {stats[k].mean():.4f}  stacked {stacked[k].mean():.4f}")
