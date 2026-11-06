import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor
from scipy.optimize import nnls
from sklearn.model_selection import KFold, train_test_split

from models import Forest, Hurdle, SplineRidge, TabMLP
from preprocess import CATS, load

VERSION = "V4"
T_START = time.time()
DEADLINE = 10 * 3600
ARM_CAP = 3 * 3600
SEEDS5, SEEDS3 = [42, 7, 2026, 11, 99], [42, 7, 2026]
FOLDS = 5
YMAX = 92
LGB_BASE = dict(learning_rate=0.02, num_leaves=31, min_child_samples=40, feature_fraction=0.7, bagging_fraction=0.8,
                bagging_freq=1, lambda_l2=1.0, verbose=-1)
LGB_VARIANTS = {"lgb": dict(objective="regression"),
                "lgb_deep": dict(objective="regression", num_leaves=127, min_child_samples=80, feature_fraction=0.5),
                "lgb_et": dict(objective="regression", extra_trees=True, num_leaves=63, learning_rate=0.03),
                "pois": dict(objective="poisson"),
                "tweedie": dict(objective="tweedie", tweedie_variance_power=1.3)}
XGB_BASE = dict(learning_rate=0.03, max_depth=6, min_child_weight=5, subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0,
                tree_method="hist", enable_categorical=True, max_cat_to_onehot=8, early_stopping_rounds=200)
XGB_VARIANTS = {"xgb": {}, "xgb_deep": dict(max_depth=9, min_child_weight=10, colsample_bytree=0.5, learning_rate=0.02)}
CAT_BASE = dict(learning_rate=0.04, depth=6, l2_leaf_reg=3, loss_function="RMSE", od_type="Iter", od_wait=200,
                thread_count=-1, verbose=0, allow_writing_files=False)
CAT_VARIANTS = {"cat": {}, "cat_d8": dict(depth=8, l2_leaf_reg=5, learning_rate=0.05)}
N = {"lgb": 20000, "xgb": 20000, "cat": 8000, "mlp": 80, "rf": 300, "et": 300, "gam": 0, "hurdle": 20000}
SMOKE = {"lgb": 30, "xgb": 30, "cat": 30, "mlp": 1, "rf": 10, "et": 10, "gam": 0, "hurdle": 30}

ARMS = [("lgb_A", "lgb", "A", False, SEEDS5), ("lgb_B", "lgb", "B", False, SEEDS5), ("lgb_C", "lgb", "C", False, SEEDS5),
        ("gam_A", "gam", "A", False, SEEDS5), ("gam_B", "gam", "B", False, SEEDS5),
        ("pois_A", "pois", "A", False, SEEDS5), ("pois_B", "pois", "B", False, SEEDS5),
        ("tweedie_A", "tweedie", "A", False, SEEDS5), ("lgb_et_A", "lgb_et", "A", False, SEEDS5),
        ("lgb_et_B", "lgb_et", "B", False, SEEDS5), ("lgb_deep_A", "lgb_deep", "A", False, SEEDS5),
        ("lgb_deep_B", "lgb_deep", "B", False, SEEDS5), ("hurdle_A", "hurdle", "A", False, SEEDS5),
        ("hurdle_B", "hurdle", "B", False, SEEDS5), ("xgb_A", "xgb", "A", False, SEEDS5),
        ("xgb_B", "xgb", "B", False, SEEDS5), ("xgb_C", "xgb", "C", False, SEEDS5),
        ("rf_A", "rf", "A", False, SEEDS3), ("et_A", "et", "A", False, SEEDS3), ("rf_B", "rf", "B", False, SEEDS3),
        ("mlp_B", "mlp", "B", True, SEEDS3), ("mlp_A", "mlp", "A", False, SEEDS3),
        ("xgb_deep_A", "xgb_deep", "A", False, SEEDS3), ("xgb_deep_B", "xgb_deep", "B", False, SEEDS3),
        ("cat_A", "cat", "A", False, SEEDS3), ("cat_B", "cat", "B", False, SEEDS3), ("cat_C", "cat", "C", False, SEEDS3),
        ("cat_B_aux", "cat", "B", True, SEEDS3), ("cat_d8_A", "cat_d8", "A", False, SEEDS3),
        ("cat_d8_B", "cat_d8", "B", False, SEEDS3)]
OUT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")

def mse(a, b):
    return float(np.mean((a - b) ** 2))

def family(kind):
    return "lgb" if kind in LGB_VARIANTS else "xgb" if kind in XGB_VARIANTS else "cat" if kind in CAT_VARIANTS else kind

def fit_model(kind, Xf, yf, wf, Xm, ym, seed, n):
    if kind in LGB_VARIANTS:
        m = lgb.LGBMRegressor(**{**LGB_BASE, **LGB_VARIANTS[kind]}, metric="l2", n_estimators=n, random_state=seed)
        m.fit(Xf, yf, sample_weight=wf, eval_set=[(Xm, ym)], callbacks=[lgb.early_stopping(200, verbose=False)])
        return m, m.best_iteration_
    if kind in XGB_VARIANTS:
        m = xgb.XGBRegressor(**{**XGB_BASE, **XGB_VARIANTS[kind]}, n_estimators=n, random_state=seed)
        m.fit(Xf, yf, sample_weight=wf, eval_set=[(Xm, ym)], verbose=False)
        return m, m.best_iteration
    if kind in CAT_VARIANTS:
        m = CatBoostRegressor(**{**CAT_BASE, **CAT_VARIANTS[kind]}, iterations=n, random_seed=seed, cat_features=CATS)
        m.fit(Xf, yf, sample_weight=wf, eval_set=(Xm, ym))
        return m, m.get_best_iteration()
    if kind == "mlp":
        m = TabMLP(CATS, seed, n).fit(Xf, yf, wf, Xm, ym)
        return m, m.best_epoch
    if kind in ("rf", "et"):
        return Forest(kind, CATS, seed, n).fit(Xf, yf, wf, Xm, ym), n
    if kind == "gam":
        m = SplineRidge(CATS, seed).fit(Xf, yf, wf, Xm, ym)
        return m, m.m.alpha_
    m = Hurdle({**LGB_BASE, "objective": "regression"}, seed, n).fit(Xf, yf, wf, Xm, ym)
    return m, m.best

def cv(kind, X, y, Xt, aux, seeds, n, folds, guard=True):
    oof, pred, seen = np.zeros(len(y)), np.zeros(len(Xt)), np.zeros(len(y))
    start = time.time()
    for s in seeds:
        t0, iters = time.time(), []
        for f, (a, b) in enumerate(KFold(FOLDS, shuffle=True, random_state=s).split(X)):
            if f >= folds:
                break
            fa, ma = train_test_split(a, test_size=0.1, random_state=s)
            Xf, yf, wf = X.iloc[fa], y[fa], np.ones(len(fa))
            if aux is not None:
                Xf, yf, wf = pd.concat([Xf, aux[0]]), np.r_[yf, aux[1]], np.r_[wf, np.ones(len(aux[1]))]
            m, it = fit_model(kind, Xf, yf, wf, X.iloc[ma], y[ma], s, n)
            oof[b] += m.predict(X.iloc[b])
            seen[b] += 1
            pred += m.predict(Xt) / (len(seeds) * folds)
            iters.append(it if isinstance(it, int) else round(float(it), 3))
            if guard and len(iters) == 1 and s == seeds[0]:
                proj = (time.time() - start) * len(seeds) * folds
                left = DEADLINE - (time.time() - T_START)
                if proj > min(ARM_CAP, left):
                    raise TimeoutError(f"projected {proj:.0f}s, cap {ARM_CAP}s, {left:.0f}s left before the deadline")
        print(f"  {kind} seed {s}: iters {iters} ({time.time() - t0:.0f}s)", flush=True)
    ok = seen > 0
    oof[ok] /= seen[ok]
    return np.clip(oof, 0, YMAX), np.clip(pred, 0, YMAX)

def str_cats(X):
    if X is None:
        return None
    X = X.copy()
    for c in CATS:
        X[c] = X[c].astype(str)
    return X

def stack(oofs, preds, y):
    A, T = np.column_stack(list(oofs.values())), np.column_stack(list(preds.values()))
    cf = np.zeros(len(y))
    for a, b in KFold(FOLDS, shuffle=True, random_state=0).split(A):
        cf[b] = A[b] @ nnls(A[a], y[a])[0]
    w = nnls(A, y)[0]
    return w, np.clip(cf, 0, YMAX), np.clip(T @ w, 0, YMAX)

def paired(a, b, y):
    d = (a - y) ** 2 - (b - y) ** 2
    return f"{d.mean():+.3f} ± {d.std() / np.sqrt(len(d)):.3f}"

def report(X, y, oof):
    print(f"OOF MSE {mse(oof, y):.3f} (rounded {mse(np.round(oof), y):.3f})")
    for name, g in [("status_last", X.status_last.astype(str).to_numpy()), ("R_90 == 0", (X.R_90 == 0).to_numpy())]:
        for k in np.unique(g):
            s = g == k
            print(f"  {name}={k}: n {s.sum()}, y mean {y[s].mean():.2f}, OOF MSE {mse(oof[s], y[s]):.2f}")
    r = y - oof
    print(f"  rows with |residual| > 30: {np.mean(np.abs(r) > 30):.3f} of rows, {np.sum(r[np.abs(r) > 30] ** 2) / np.sum(r ** 2):.3f} of SSE")

def write(pred, ids, oofs, preds, X):
    sub = pd.DataFrame({"PropertyID_test": ids, "Pred": np.round(pred).astype(int)})
    assert len(sub) == 24318 and sub.PropertyID_test.is_unique and sub.Pred.between(0, YMAX).all()
    path = OUT / "submission.csv"
    sub.to_csv(path, index=False)
    extra = OUT / "extra" / VERSION
    extra.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"PropertyID": X.index, **oofs}).to_csv(extra / "oof.csv", index=False)
    pd.DataFrame({"PropertyID": ids, **preds}).to_csv(extra / "pred.csv", index=False)
    return path

if __name__ == "__main__":
    views, y, y2, ids = load()
    print(f"view C {views['C'][0].shape[1]} features")
    data = {}
    for _, kind, view, _, _ in ARMS:
        if (family(kind), view) not in data:
            data[family(kind), view] = tuple(str_cats(x) for x in views[view]) if family(kind) == "cat" else views[view]
    XB = views["B"][0]
    print(f"load {time.time() - T_START:.0f}s", flush=True)
    p = []
    for name, kind, view, use_aux, _ in ARMS:
        X, Xt, X2 = data[family(kind), view]
        p.append(cv(kind, X, y, Xt, (X2, y2) if use_aux else None, SEEDS3[:1], SMOKE[family(kind)], 1, guard=False)[1])
    write(np.mean(p, 0), ids, {}, {}, XB)
    print(f"smoke ok ({time.time() - T_START:.0f}s)", flush=True)
    oofs, preds = {}, {}
    for name, kind, view, use_aux, seeds in ARMS:
        X, Xt, X2 = data[family(kind), view]
        try:
            t1 = time.time()
            oofs[name], preds[name] = cv(kind, X, y, Xt, (X2, y2) if use_aux else None, seeds, N[family(kind)], FOLDS)
        except Exception as e:
            print(f"{name} skipped: {e!r}", flush=True)
            continue
        vs = f", vs lgb_A {paired(oofs[name], oofs['lgb_A'], y)}" if "lgb_A" in oofs and name != "lgb_A" else ""
        print(f"{name}: OOF MSE {mse(oofs[name], y):.3f}{vs} ({time.time() - t1:.0f}s, elapsed {(time.time() - T_START) / 3600:.2f} h)", flush=True)

        w, oof, pred = stack(oofs, preds, y)
        write(pred, ids, {**oofs, "stack": oof}, {**preds, "stack": pred}, XB)
        print(f"  stack of {len(oofs)} arms so far: OOF {mse(oof, y):.3f}", flush=True)
    members = list(oofs)
    assert members, "no arm finished"
    E = pd.DataFrame({k: oofs[k] - y for k in members})
    C = E.corr()
    eq = np.mean([oofs[k] for k in members], 0)

    w, oof, pred = stack(oofs, preds, y)
    print("arm summary (OOF, error corr with lgb_A, stack weight):")
    for k, wk in zip(members, w):
        print(f"  {k:12s} {mse(oofs[k], y):8.3f}  {C.loc[k, 'lgb_A'] if 'lgb_A' in C else np.nan:6.3f}  {wk:.3f}")
    print(f"equal-weight mean of {len(members)} arms: OOF {mse(eq, y):.3f}")
    if "lgb_A" in oofs:
        print(f"stack (cross-fitted weights) vs lgb_A {paired(oof, oofs['lgb_A'], y)}")
    oofs["stack"], preds["stack"] = oof, pred
    report(XB, y, oof)
    path = write(pred, ids, oofs, preds, XB)
    print(f"total {(time.time() - T_START) / 3600:.2f} h")
    print(f"SUBMIT: {path}")
