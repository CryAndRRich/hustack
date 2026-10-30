import time

import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold

from preprocess import TARGET, strata

N_SPLITS, N_REPEATS, SEED = 5, 20, 0


def splits(tr, n_splits=N_SPLITS, n_repeats=N_REPEATS, seed=SEED):
    rs = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    s = strata(tr).values

    lab = np.array([v if (s == v).sum() >= n_splits else v.split("_")[0] for v in s])
    return list(rs.split(np.zeros(len(tr)), lab))


def oof(fit_predict, tr, folds, n_repeats=N_REPEATS):
    out = np.full((n_repeats, len(tr)), np.nan)
    per = len(folds) // n_repeats
    for k, (a, b) in enumerate(folds):
        out[k // per, b] = fit_predict(tr.iloc[a].reset_index(drop=True), tr.iloc[b].reset_index(drop=True))
    assert np.isfinite(out).all()
    return out


def run(models, tr, folds, n_repeats=N_REPEATS):
    y = tr[TARGET].values.astype(float)
    res = {}
    for name, fp in models.items():
        t0 = time.time()
        P = oof(fp, tr, folds, n_repeats)
        per_rep = np.sqrt(((P - y) ** 2).mean(axis=1))
        res[name] = {"P": P, "rmse": per_rep.mean()}
        print(f"{name:28s} rmse {per_rep.mean():9.0f} ± {per_rep.std():6.0f}  ({time.time() - t0:5.1f}s)", flush=True)
    return res
