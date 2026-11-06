import time
import warnings
from pathlib import Path

import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import nnls

from preprocess import load

VERSION = "V2"
T_START = time.time()
K, REPEATS, BOOT = 5, 1, 300
W_CLIP = (0.05, 20.0)
LGB_PARAMS = dict(learning_rate=0.04, num_leaves=31, min_child_samples=10, feature_fraction=0.5, bagging_fraction=0.8,
                  bagging_freq=1, lambda_l2=1.0, n_jobs=-1, verbose=-1)
CB_PARAMS = dict(learning_rate=0.06, depth=6, l2_leaf_reg=3.0, loss_function="RMSE", thread_count=-1, verbose=0)
N_MAX, STOP = 6000, 200
ARMS = {"lgb_raw": ("lgb", False, True), "lgb_log": ("lgb", True, True), "cat_raw": ("cat", False, True),
        "lgb_raw_v1": ("lgb", False, False)}
warnings.filterwarnings("ignore", message="X does not have valid feature names")
OUT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")


def group_folds(groups, seed):
    u = np.unique(groups)
    rank = dict(zip(np.random.default_rng(seed).permutation(u), range(len(u))))
    return np.array([rank[g] % K for g in groups])


def r2(y, p, w=None):
    w = np.ones_like(y) if w is None else w
    mu = np.average(y, weights=w)
    return 1 - np.average((y - p) ** 2, weights=w) / np.average((y - mu) ** 2, weights=w)


def adversarial_weights(X, is_test, groups):
    f, p = group_folds(groups, 0), np.zeros(len(X))
    for k in range(K):
        m = lgb.LGBMClassifier(**LGB_PARAMS, n_estimators=500, random_state=0)
        m.fit(X[f != k], is_test[f != k])
        p[f == k] = m.predict_proba(X[f == k])[:, 1]
    pos = np.argsort(np.argsort(p))
    auc = (pos[is_test].sum() - is_test.sum() * (is_test.sum() - 1) / 2) / (is_test.sum() * (~is_test).sum())
    w = np.clip(p / (1 - p) * (~is_test).sum() / is_test.sum(), *W_CLIP)[~is_test]
    w /= w.mean()
    print(f"adversarial AUC {auc:.4f}; train weights: ESS {w.sum() ** 2 / (w ** 2).sum():.0f} of {len(w)}, "
          f"max {w.max():.2f}, share clipped high {np.mean(w >= w.max() - 1e-9):.3f}", flush=True)
    return w


def fit_predict(arm, X, y, core, fa, va, Xt, seed, n=None):
    kind, log, use_hypo = ARMS[arm]
    fa = fa if use_hypo else fa & core
    t = np.log1p(y) if log else y
    if kind == "lgb":
        m = lgb.LGBMRegressor(**LGB_PARAMS, n_estimators=n or N_MAX, random_state=seed)
        m.fit(X[fa], t[fa], eval_set=[(X[va], t[va])], callbacks=[lgb.early_stopping(STOP, verbose=False)])
        it = m.best_iteration_
    else:
        m = cb.CatBoostRegressor(**CB_PARAMS, iterations=n or N_MAX, random_seed=seed, od_type="Iter", od_wait=STOP)
        m.fit(X[fa], t[fa], eval_set=(X[va], t[va]), use_best_model=True)
        it = m.get_best_iteration()
    pv, pt = m.predict(X[va]), m.predict(Xt)
    if log:
        pv, pt = np.expm1(pv), np.expm1(pt)
    return pv, pt, it, m


def cross_fit(X, y, core, Xt, groups, n=None, repeats=REPEATS, folds=K):
    oof = {a: np.zeros((repeats, len(y))) for a in ARMS}
    test = {a: np.zeros(len(Xt)) for a in ARMS}
    iters = {a: [] for a in ARMS}
    last = {}
    for r in range(repeats):
        f = group_folds(groups, 100 + r)
        for k in range(folds):
            fa, va = (f != k) | ~core, (f == k) & core
            for a in ARMS:
                pv, pt, it, last[a] = fit_predict(a, X, y, core, fa, va, Xt, 100 * r + k, n)
                oof[a][r, va] = pv
                test[a] += pt / (repeats * folds)
                iters[a].append(it)
            print(f"  repeat {r} fold {k} done ({time.time() - T_START:.0f}s)", flush=True)
    return oof, test, iters, last


def blend_weights(P, y, w):
    s = np.sqrt(w)
    b, _ = nnls(P * s[:, None], y * s)
    return b


def paired(oof, y, w, groups, ref):
    rng = np.random.default_rng(0)
    idx = pd.Series(np.arange(len(y))).groupby(groups).indices
    keys = list(idx)
    d = {a: [] for a in oof if a != ref}
    for _ in range(BOOT):
        b = np.concatenate([idx[keys[i]] for i in rng.integers(0, len(keys), len(keys))])
        r0 = r2(y[b], oof[ref][b], w[b])
        for a in d:
            d[a].append(r2(y[b], oof[a][b], w[b]) - r0)
    for a, v in d.items():
        print(f"  {a:13s} vs {ref}: weighted R2 {np.mean(v):+.4f} +/- {np.std(v):.4f}")


def report(name, p, y, w, inacc, comp):
    print(f"  {name:13s} R2 {r2(y, p):.4f} | weighted {r2(y, p, w):.4f} | competition rows {r2(y[comp], p[comp]):.4f} "
          f"weighted {r2(y[comp], p[comp], w[comp]):.4f} | ASA=0 {r2(y[inacc], p[inacc]):.4f} "
          f"| ASA>0 {r2(y[~inacc], p[~inacc]):.4f} | mean pred {p.mean():.4f}", flush=True)


def write(names, pred, extra):
    sub = pd.DataFrame({"MOFname": names, "Prediction": np.clip(pred, 0, None)})
    assert len(sub) == 2000 and sub.MOFname.is_unique and np.isfinite(sub.Prediction).all()
    path = OUT / "submission.csv"
    sub.to_csv(path, index=False)
    d = OUT / "extra" / VERSION
    d.mkdir(parents=True, exist_ok=True)
    for name, df in extra.items():
        df.to_csv(d / f"{name}.csv", index=False)
    return path


if __name__ == "__main__":
    a, features = load()
    lab = a.y.notna().to_numpy()
    F = a[features].to_numpy(float)
    X, Xt = F[lab], F[~lab]
    y, groups, names = a.y[lab].to_numpy(), a.group[lab].to_numpy(), a.MOFname[~lab].to_numpy()
    src = a.ext.to_numpy()
    core = src[lab] < 2
    keep = src < 2
    w = adversarial_weights(F[keep], ~lab[keep], a.group.to_numpy()[keep])
    yc, gc, ext = y[core], groups[core], src[lab][core] == 1
    comp, inacc = ~ext, a.inaccessible.to_numpy()[lab][core] == 1
    print(f"scored CoRE rows {core.sum()}, hypothetical fit-only rows {(~core).sum()}; weighted share of ASA=0 "
          f"{np.average(inacc, weights=w):.3f} (test {a.inaccessible[~lab].mean():.3f})", flush=True)
    _, sm, _, _ = cross_fit(X, y, core, Xt, groups, n=20, repeats=1, folds=1)
    write(names, sm["lgb_raw"], {})
    print(f"smoke ok ({time.time() - T_START:.0f}s)", flush=True)
    oof, test, iters, last = cross_fit(X, y, core, Xt, groups)
    oof = {k: v[:, core] for k, v in oof.items()}
    print("iterations (median): " + ", ".join(f"{k} {int(np.median(v))}" for k, v in iters.items()))
    print("per repeat weighted R2: " + ", ".join(f"{k} " + "/".join(f"{r2(yc, o, w):.4f}" for o in v) for k, v in oof.items()))
    oof = {k: v.mean(0) for k, v in oof.items()}
    for k, v in oof.items():
        report(k, v, yc, w, inacc, comp)
    P, T = np.column_stack(list(oof.values())), np.column_stack(list(test.values()))
    b = blend_weights(P, yc, w)
    print("NNLS blend weights: " + ", ".join(f"{k} {v:.3f}" for k, v in zip(oof, b)) + f" (sum {b.sum():.3f})")
    oof["blend"], test["blend"] = P @ b, T @ b
    oof["mean"], test["mean"] = P.mean(1), T.mean(1)
    for k in ("blend", "mean"):
        report(k, oof[k], yc, w, inacc, comp)
    paired(oof, yc, w, gc, "lgb_raw_v1")
    imp = pd.Series(last["lgb_raw"].booster_.feature_importance("gain"), index=features).sort_values(ascending=False)
    print("lgb_raw gain top 25: " + ", ".join(f"{c} {v / imp.sum():.3f}" for c, v in imp.head(25).items()))
    print("test mean pred: " + ", ".join(f"{k} {v.mean():.4f}" for k, v in test.items()))
    extra = {"oof": pd.DataFrame({"MOFname": a.MOFname[lab].to_numpy()[core], "ext": ext.astype(int), "y": yc, "w": w, **oof}),
             "test_pred": pd.DataFrame({"MOFname": names, **test})}
    path = write(names, test["blend"], extra)
    print(f"total {(time.time() - T_START) / 60:.1f} min")
    print(f"SUBMIT: {path}")
