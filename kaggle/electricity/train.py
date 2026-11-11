import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold

from preprocess import ID, TARGET, build, load

VERSION = "V2"
OUT = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path(".")
FULL = dict(repeats=3, folds=5, rounds=6000)
SMOKE = dict(repeats=1, folds=2, rounds=200)
CHECKPOINTS = (1000, 2000, 3000, 4000, 5000, 6000)
PARAMS = dict(objective="regression", learning_rate=0.01, num_leaves=31, min_child_samples=20, feature_fraction=0.7,
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, n_jobs=-1)
ARMS = {"v1": False, "grid": True}


def fit_predict(Xa, ya, Xb, rounds, seed):
    m = lgb.train({**PARAMS, "seed": seed}, lgb.Dataset(Xa, ya), num_boost_round=rounds)
    cps = [c for c in CHECKPOINTS if c <= rounds] or [rounds]
    return {c: m.predict(Xb, num_iteration=c) for c in cps}


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def cv(X, y, repeats, folds, rounds):
    oof = {}
    for k, (i, j) in enumerate(RepeatedKFold(n_splits=folds, n_repeats=repeats, random_state=0).split(X)):
        for c, p in fit_predict(X.iloc[i], y[i], X.iloc[j], rounds, seed=k).items():
            oof.setdefault(c, np.zeros((repeats, len(y))))[k // folds, j] = p
    scores = {c: np.array([rmse(o[r], y) for r in range(repeats)]) for c, o in oof.items()}
    return scores, {c: o.mean(0) for c, o in oof.items()}


def run(tr, te, repeats, folds, rounds, smoke):
    t0 = time.time()
    y = tr[TARGET].to_numpy()
    mats, res = {}, {}
    print(f"{'smoke' if smoke else 'full'}: {repeats} x {folds}-fold CV, up to {rounds} rounds")
    for arm, grid in ARMS.items():
        mats[arm] = build(tr, te, grid)
        res[arm] = cv(mats[arm][0], y, repeats, folds, rounds)
        for c, s in res[arm][0].items():
            print(f"  {arm:5s} rounds {c:5d}: OOF RMSE {s.mean():.5f} (repeat sd {s.std():.5f})")
        print(f"  {arm}: {mats[arm][0].shape[1]} columns ({time.time() - t0:.0f}s)", flush=True)
    best = {arm: min(s, key=lambda c: s[c].mean()) for arm, (s, _) in res.items()}
    d = res["grid"][0][best["grid"]] - res["v1"][0][best["v1"]]
    print(f"  grid - v1 at their best rounds: {d.mean():+.5f} per repeat {np.round(d, 5)}")
    pick = "grid" if d.max() < 0 else "v1"
    resid = y - res[pick][1][best[pick]]
    print(f"  {pick} OOF residual quantiles (1, 5, 50, 95, 99 %): {np.round(np.quantile(resid, [.01, .05, .5, .95, .99]), 3)}; "
          f"|resid| > 1.5: {(np.abs(resid) > 1.5).sum()} rows")
    if smoke:
        return
    extra = OUT / "extra" / VERSION
    extra.mkdir(parents=True, exist_ok=True)
    for arm, (X, Xt) in mats.items():
        p = np.mean([fit_predict(X, y, Xt, best[arm], seed=100 + s)[best[arm]] for s in range(3)], 0)
        p = np.clip(p, 0, None)
        assert len(p) == len(te) and np.isfinite(p).all()
        sub = pd.DataFrame({ID: te[ID], TARGET: p})
        sub.to_csv(extra / f"{arm}.csv", index=False)
        if arm == pick:
            sub.to_csv(OUT / "submission.csv", index=False)
        print(f"  {arm}: {best[arm]} rounds, 3 seeds, test mean {p.mean():.4f} -> {extra / arm}.csv")
    print(f"submission: {pick}, OOF RMSE {res[pick][0][best[pick]].mean():.5f}")
    print(f"SUBMIT: {OUT / 'submission.csv'}")


if __name__ == "__main__":
    t0 = time.time()
    tr, te = load()
    run(tr, te, smoke=True, **SMOKE)
    print("smoke ok")
    run(tr, te, smoke=False, **FULL)
    print(f"total {(time.time() - t0) / 60:.1f} min")
