import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog

from runtime import log


def lp_weights(P, y):
    n, k = P.shape
    c = np.r_[np.zeros(k), np.ones(n)]
    eye = sparse.identity(n, format="csr")
    A_ub = sparse.vstack([sparse.hstack([sparse.csr_matrix(P), -eye]), sparse.hstack([sparse.csr_matrix(-P), -eye])])
    b_ub = np.r_[y, -y]
    A_eq = sparse.csr_matrix(np.r_[np.ones(k), np.zeros(n)][None])
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=[1.0], bounds=[(0, None)] * (k + n), method="highs")
    assert res.status == 0, res.message
    return res.x[:k]


def nested_blend(P, y, fold):
    out = np.zeros(len(y))
    for f in np.unique(fold):
        tr = fold != f
        out[~tr] = P[~tr] @ lp_weights(P[tr], y[tr])
    return out


def paired_se(err_a, err_b, n_boot, seed):
    d = np.abs(err_a) - np.abs(err_b)
    rng = np.random.default_rng(seed)
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)]
    return float(d.mean()), float(np.std(boots))


def fairness(y, pred, groups):
    report = {}
    for col, g in groups.items():
        per = pd.Series(np.abs(y - pred)).groupby(g.values).mean()
        report[col] = {"mae": per.round(3).to_dict(), "max_gap": float(per.max() - per.min())}
        log("fairness", col, report[col])
    return report
