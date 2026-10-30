import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy.sparse as sp
import xgboost as xgb
from catboost import CatBoostRegressor
from scipy.spatial import cKDTree
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_predict

from preprocess import TE_KEYS, features, load
from text import choose_alpha, ridge_features, sparse_designs

SEED = 42
FOLDS = 5
INNER = 5
ALPHA = 5.0
SIGMA_KM, CUT_KM = 0.6, 1.8
LO, HI = np.log1p(20), np.log1p(2000)
LGB_PARAMS = dict(objective="regression", learning_rate=0.03, num_leaves=63, min_child_samples=20,
                  feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  n_estimators=10000, verbose=-1)
LGB_SEEDS = [42, 7, 2026]
CAT_PARAMS = dict(iterations=6000, learning_rate=0.06, depth=7, l2_leaf_reg=3, border_count=128,
                  od_type="Iter", od_wait=200, thread_count=-1, verbose=0, random_seed=SEED,
                  allow_writing_files=False)
XGB_PARAMS = dict(n_estimators=10000, learning_rate=0.03, max_depth=8, min_child_weight=3, subsample=0.8,
                  colsample_bytree=0.6, reg_lambda=1.0, tree_method="hist", enable_categorical=True,
                  max_cat_to_onehot=8, early_stopping_rounds=200, random_state=SEED)
CONFIGS = {"V1": ["base", "te"], "+mined": ["base", "te", "mined"], "+geo": ["base", "te", "geo", "te_geo"],
           "+text": ["base", "te", "ridge"], "full": ["base", "te", "mined", "geo", "te_geo", "ridge"]}
OUT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")


def key_of(keys, cols):
    return pd.Series(pd.MultiIndex.from_frame(keys[cols].astype(str)).to_flat_index())


def te_apply(k_fit, y_fit, k_apply, prior):
    g = pd.DataFrame({"k": k_fit.values, "y": y_fit}).groupby("k").y.agg(["sum", "count"])
    s, c = k_apply.map(g["sum"]).fillna(0).values, k_apply.map(g["count"]).fillna(0).values
    return (s + ALPHA * prior) / (c + ALPHA), c


def enc_keys(k_fit, y_fit, k_apply):
    out, prior = {}, y_fit.mean()
    for cols in TE_KEYS:
        name = "te_" + "_".join(cols)
        out[name], out[name + "_n"] = te_apply(key_of(k_fit, cols), y_fit, key_of(k_apply, cols), prior)
    return out


def kernel(cells):
    xy = cells[["geo_x", "geo_y"]].values
    assert np.isfinite(xy).all()
    d = cKDTree(xy).sparse_distance_matrix(cKDTree(xy), CUT_KM, output_type="coo_matrix")
    W = sp.csr_matrix((np.exp(-0.5 * (d.data / SIGMA_KM) ** 2), (d.row, d.col)), shape=d.shape).tolil()
    W.setdiag(1.0)
    return W.tocsr()


def make_enc_space(W):
    def enc_space(k_fit, y_fit, k_apply):
        prior, ci, ca = y_fit.mean(), k_fit.ci.values, k_apply.ci.values
        out = {}
        for name, sel in [("sp_all", np.ones(len(ci), bool)), ("sp_entire", (k_fit.room_type == "Entire home/apt").values),
                          ("sp_private", (k_fit.room_type == "Private room").values)]:
            ok = sel & (ci >= 0)
            s = W @ np.bincount(ci[ok], y_fit[ok], minlength=W.shape[0])
            c = W @ np.bincount(ci[ok], minlength=W.shape[0]).astype(float)
            v = (s + ALPHA * prior) / (c + ALPHA)
            out[name] = np.where(ca >= 0, v[np.maximum(ca, 0)], np.nan)
            if name == "sp_all":
                out["sp_all_n"] = np.where(ca >= 0, c[np.maximum(ca, 0)], np.nan)
        return out
    return enc_space


def nested(enc, k_fit, y_fit, k_apps, inner):
    parts = [(b, enc(k_fit.iloc[a], y_fit[a], k_fit.iloc[b])) for a, b in inner]
    fit = pd.DataFrame({n: np.zeros(len(k_fit)) for n in parts[0][1]})
    for b, p in parts:
        for n, v in p.items():
            fit.iloc[b, fit.columns.get_loc(n)] = v
    return fit, [pd.DataFrame(enc(k_fit, y_fit, k)) for k in k_apps]


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def to_cat(X, keys):
    X = X.copy()
    for c in X.columns[X.dtypes == "category"]:
        X[c] = X[c].astype(str)
    X["geo_cell_id"], X["host_id_id"] = keys.geo_cell.astype(str).values, keys.host_id.astype(str).values
    return X


def build_folds(blocks, keys, text, cells, y):
    n = len(y)
    keys = keys.assign(ci=keys.geo_cell.map(pd.Series(np.arange(len(cells)), index=cells.index)).fillna(-1).astype(int))
    enc_space = make_enc_space(kernel(cells))
    designs = sparse_designs(text, blocks["base"])
    K_tr, K_te = keys.iloc[:n].reset_index(drop=True), keys.iloc[n:].reset_index(drop=True)
    te_rows = np.arange(n, len(keys))
    folds, alphas = [], None
    for f, (a, b) in enumerate(KFold(FOLDS, shuffle=True, random_state=SEED).split(np.zeros(n))):
        t0 = time.time()
        if alphas is None:
            print("ridge alpha, chosen on fold-0 fit rows:")
            alphas = {k: choose_alpha(M[a], y[a], SEED + 2) for k, M in designs.items()}
        inner = list(KFold(INNER, shuffle=True, random_state=SEED + 1).split(a))
        Ka = K_tr.iloc[a].reset_index(drop=True)
        enc = {"te": nested(enc_keys, Ka, y[a], [K_tr.iloc[b].reset_index(drop=True), K_te], inner),
               "te_geo": nested(enc_space, Ka, y[a], [K_tr.iloc[b].reset_index(drop=True), K_te], inner)}
        rf, (rb, rt) = ridge_features(designs, alphas, a, y[a], [b, te_rows], inner)
        enc["ridge"] = (pd.DataFrame(rf), [pd.DataFrame(rb), pd.DataFrame(rt)])
        parts = {}
        for name, X in blocks.items():
            parts[name] = (X.iloc[a].reset_index(drop=True), [X.iloc[b].reset_index(drop=True),
                                                              X.iloc[n:].reset_index(drop=True)])
        parts.update(enc)
        folds.append((a, b, parts))
        print(f"fold {f} features: ridge val " + ", ".join(f"{k} {rmse(rb[k], y[b]):.4f}" for k in rb)
              + f" ({time.time() - t0:.0f}s)")
    return folds, K_tr, K_te


def assemble(parts, names):
    fit = pd.concat([parts[k][0] for k in names], axis=1)
    val = pd.concat([parts[k][1][0] for k in names], axis=1)
    tst = pd.concat([parts[k][1][1] for k in names], axis=1)
    return fit, val, tst


def run_model(kind, folds, names, y, n_test, K_tr, K_te, seed=SEED):
    t0 = time.time()
    oof, pred, iters = np.zeros(len(y)), np.zeros(n_test), []
    for a, b, parts in folds:
        Xa, Xb, Xt = assemble(parts, names)
        if kind == "lgb":
            m = lgb.LGBMRegressor(**LGB_PARAMS, random_state=seed)
            m.fit(Xa, y[a], eval_set=[(Xb, y[b])], callbacks=[lgb.early_stopping(200, verbose=False)])
            iters.append(m.best_iteration_)
        elif kind == "xgb":
            m = xgb.XGBRegressor(**XGB_PARAMS)
            m.fit(Xa, y[a], eval_set=[(Xb, y[b])], verbose=False)
            iters.append(m.best_iteration)
        else:
            Xa, Xb, Xt = to_cat(Xa, K_tr.iloc[a]), to_cat(Xb, K_tr.iloc[b]), to_cat(Xt, K_te)
            cat_cols = [c for c in Xa.columns if Xa[c].dtype == object]
            m = CatBoostRegressor(**CAT_PARAMS, cat_features=cat_cols)
            m.fit(Xa, y[a], eval_set=(Xb, y[b]), use_best_model=True)
            iters.append(m.get_best_iteration())
        oof[b] = np.clip(m.predict(Xb), LO, HI)
        pred += np.clip(m.predict(Xt), LO, HI) / len(folds)
    print(f"  {kind:4s} seed {seed:5d} [{'+'.join(names)}]: OOF {rmse(oof, y):.5f}  iters {iters} ({time.time() - t0:.0f}s)")
    return oof, pred


def write(pred, ids, path):
    sub = pd.DataFrame({"id": ids, "price": np.expm1(pred).round(2)})
    assert sub.price.between(20, 2000).all()
    path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(path, index=False)


def run():
    tr, te, geo = load()
    blocks, keys, text, cells, y, ids = features(tr, te, geo)
    folds, K_tr, K_te = build_folds(blocks, keys, text, cells, y)
    print(f"constant OOF {rmse(y.mean(), y):.5f}")

    print("ablation (LightGBM, one V2 block at a time):")
    arms = {}
    for label, names in CONFIGS.items():
        arms["lgb_" + label] = run_model("lgb", folds, names, y, len(ids), K_tr, K_te)
    full = CONFIGS["full"]
    print("full-feature arms:")
    seeds = [arms["lgb_full"]] + [run_model("lgb", folds, full, y, len(ids), K_tr, K_te, s) for s in LGB_SEEDS[1:]]
    arms["lgb"] = (np.mean([s[0] for s in seeds], axis=0), np.mean([s[1] for s in seeds], axis=0))
    print(f"  lgb {len(seeds)}-seed mean: OOF {rmse(arms['lgb'][0], y):.5f}")
    arms["xgb"] = run_model("xgb", folds, full, y, len(ids), K_tr, K_te)
    arms["cat"] = run_model("cat", folds, full, y, len(ids), K_tr, K_te)
    for k in ["ridge_text", "ridge_full"]:
        oof = np.zeros(len(y))
        for a, b, parts in folds:
            oof[b] = parts["ridge"][1][0][k].values
        test = np.mean([parts["ridge"][1][1][k].values for _, _, parts in folds], axis=0)
        arms[k] = (np.clip(oof, LO, HI), np.clip(test, LO, HI))
        print(f"  {k}: OOF {rmse(arms[k][0], y):.5f}")

    stack_arms = ["lgb", "xgb", "cat", "ridge_full", "ridge_text"]
    O = np.column_stack([arms[k][0] for k in stack_arms])
    T = np.column_stack([arms[k][1] for k in stack_arms])
    st = LinearRegression(positive=True)
    honest = np.clip(cross_val_predict(st, O, y, cv=KFold(5, shuffle=True, random_state=SEED + 9)), LO, HI)
    st.fit(O, y)
    print("stack weights: " + ", ".join(f"{k} {w:.3f}" for k, w in zip(stack_arms, st.coef_))
          + f", intercept {st.intercept_:.3f}")
    blend = np.clip(st.predict(T), LO, HI)
    print(f"stack OOF (nested) {rmse(honest, y):.5f} | mean of lgb/xgb/cat "
          f"{rmse(O[:, :3].mean(axis=1), y):.5f}")
    for rt in tr.room_type.unique():
        s = (tr.room_type == rt).values
        print(f"  {rt:16s} n={s.sum():6d} stack {rmse(honest[s], y[s]):.5f}  V1-config {rmse(arms['lgb_V1'][0][s], y[s]):.5f}")

    write(arms["lgb"][1], ids, OUT / "extra" / "submission_lgb.csv")
    write(blend, ids, OUT / "submission.csv")
    print(f"SUBMIT: submission.csv  (V2 stack of lgb/xgb/cat/ridge, nested stack OOF {rmse(honest, y):.5f})")


if __name__ == "__main__":
    run()
