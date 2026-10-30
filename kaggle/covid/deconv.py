import numpy as np
import pandas as pd

from preprocess import FEATS, TARGET

WIN = 7
LAMS = (1e-3, 1e-4, 1e-5, 1e-6)
CLI = ("cli", "ili", "hh_cmnty_cli", "nohh_cmnty_cli")
LAM_CLI = 1e-5


def _means(n):
    m = n + WIN - 1
    A = np.zeros((n, m))
    for i in range(n):
        A[i, i:i + WIN] = 1.0 / WIN
    return A


def _solver(n, order):
    A = _means(n)
    m = A.shape[1]
    D = np.diff(np.eye(m), order, axis=0)
    K = np.block([[2 * D.T @ D, A.T], [A, np.zeros((n, n))]])
    return np.linalg.inv(K)[:m, m:]


def _trend_solver(n, lam):
    A = _means(n)
    m = A.shape[1]
    D = np.diff(np.eye(m), 2, axis=0)
    S = np.linalg.inv(np.eye(m) + D.T @ D / lam)
    K = np.block([[2 * (np.eye(m) - S), A.T], [A, np.zeros((n, n))]])
    P = np.linalg.inv(K)[:m, m:]
    return P, S @ P


def deconv_features(panel, order=2, min_days=14):
    cache = {}

    def solve(S, key):
        n = len(S)
        if (n, key) not in cache:
            cache[(n, key)] = _solver(n, order) if key is None else _trend_solver(n, key)
        return cache[(n, key)] @ S if key is None else [M @ S for M in cache[(n, key)]]

    dc_cols = [f"xtp_{k}" for k in range(1, WIN + 1)] + [f"x{lag}_{f}" for f in FEATS for lag in (0, WIN)]
    kf_cols = ([f"{c}_{lam:g}" for lam in LAMS for c in ("tr", "ep")]
               + [f"en{k}_{f}" for f in CLI for k in (0, WIN, 2 * WIN)])
    dc, kf = (np.full((len(panel), len(c)), np.nan) for c in (dc_cols, kf_cols))
    ci = [FEATS.index(f) for f in CLI]
    for _, p in panel.groupby("seg"):
        tp, sv = p[TARGET].to_numpy(), p[FEATS].to_numpy()
        for r, t in zip(p.index, p["t"]):
            if t < min_days:
                continue
            xt, xs = solve(tp[:t], None), solve(sv[:t + 1], None)
            dc[r] = np.r_[xt[t + WIN - 1 - np.arange(1, WIN + 1)], np.c_[xs[t + WIN - 1], xs[t - 1]].ravel()]
            row = []
            for lam in LAMS:
                x, s = solve(tp[:t], lam)
                i = t + WIN - 1
                row += [(2 * s[i - 1] - s[i - 2] - s[i - WIN]) / WIN, (x[i - WIN] - s[i - WIN]) / WIN]
            x, s = solve(sv[:t + 1, ci], LAM_CLI)
            e = x - s
            row += list(e[t + WIN - 1 - np.array([0, WIN, 2 * WIN])].T.ravel())
            kf[r] = row
    return pd.DataFrame(dc, index=panel.index, columns=dc_cols), pd.DataFrame(kf, index=panel.index, columns=kf_cols)
