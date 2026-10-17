import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from scipy.optimize import nnls
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from families import GPU_FAMILIES, KNN_COLS, Design, cuda_available, free_gpu, predict_family
from preprocess import (CAT_COLS, HIER_TE_COLS, LOW_CARD_COLS, POOL_CAT, POOL_FEATURES, POOL_LOC_CAT, POOL_NUM,
                        POOL_REF_SOURCES, V2_EXT_COLS, V2_NUM_COLS, add_pool_features, build_features,
                        encode_train_test, get_out_dir, load_data, load_pool)

VERSION = "V4"
FEATURES = V2_NUM_COLS + V2_EXT_COLS + POOL_FEATURES
TE_USE = HIER_TE_COLS
N_FOLDS, N_REPEATS, INNER_FOLDS = 5, 2, 5
N_SEEDS = 1
CORE = ["hier_te", "ridge", "lgbm", "lgbm_log", "catboost", "catboost_te"]
EXTRA = ["xgb_log", "hgb", "forest", "knn", "mlp", "svr", "huber", "pls", "krr",
         "tabpfn", "tabicl", "tabdpt", "realmlp", "tabm"]
MODELS = CORE + EXTRA
LGB_PARAMS = dict(
    n_estimators=1500, learning_rate=0.02, num_leaves=15, min_child_samples=15, subsample=0.8,
    subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1,
)
POOL_PARAMS = dict(
    iterations=2500, learning_rate=0.06, depth=8, loss_function="RMSE", verbose=0,
    allow_writing_files=False, thread_count=-1,
)
POOL_HOLDOUT = 0.1
CAT_PARAMS = dict(
    iterations=1500, learning_rate=0.04, depth=6, loss_function="RMSE", verbose=0,
    allow_writing_files=False, thread_count=-1,
)


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def pool_predictions(pool):
    ext = pool[pool["source"] != "competition"].reset_index(drop=True)
    comp = pool[pool["source"] == "competition"].reset_index(drop=True)
    y = np.log(ext["price"].to_numpy(dtype=float))
    hold = np.random.default_rng(0).random(len(ext)) < POOL_HOLDOUT
    out = pd.DataFrame({"row_key": comp["row_key"]})
    for name, cats in [("log_pool_px", POOL_CAT), ("log_pool_px_loc", POOL_LOC_CAT)]:
        cols = POOL_NUM + cats
        model = CatBoostRegressor(**POOL_PARAMS, random_seed=0, cat_features=cats)
        model.fit(ext.loc[~hold, cols], y[~hold])
        hold_rmse = rmse(model.predict(ext.loc[hold, cols]), y[hold])
        preds = []
        for s in POOL_REF_SOURCES:
            X = comp[cols].copy()
            X["source"] = s
            preds.append(model.predict(X))
        out[name] = np.mean(preds, axis=0)
        print(f"pool model {name}: random-holdout log-RMSE {hold_rmse:.4f} on {int(hold.sum())} external rows")
    return out


def lgb_frame(F, te, categories):
    X = pd.concat([F[FEATURES], te[TE_USE]], axis=1)
    for c in LOW_CARD_COLS:
        X[c] = pd.Categorical(F[c], categories=categories[c])
    return X


def fit_predict(name, Fa, ya, Fb, te_a, te_b, seed, categories):
    if name == "hier_te":
        return te_b["te_community"].to_numpy()
    if name == "ridge":
        cols = FEATURES + TE_USE
        pre = ColumnTransformer([
            ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), LOW_CARD_COLS),
        ])
        model = make_pipeline(pre, RidgeCV(alphas=np.logspace(-2, 3, 21)))
        Xa = pd.concat([Fa[FEATURES + LOW_CARD_COLS], te_a[TE_USE]], axis=1)
        Xb = pd.concat([Fb[FEATURES + LOW_CARD_COLS], te_b[TE_USE]], axis=1)
        return model.fit(Xa, ya).predict(Xb)
    if name in ("lgbm", "lgbm_log"):
        Xa, Xb = lgb_frame(Fa, te_a, categories), lgb_frame(Fb, te_b, categories)
        preds = []
        for s in range(N_SEEDS):
            model = lgb.LGBMRegressor(**LGB_PARAMS, random_state=100 * seed + s)
            if name == "lgbm":
                preds.append(model.fit(Xa, ya).predict(Xb))
                continue
            model.fit(Xa, np.log(ya))
            smear = np.mean(np.exp(np.log(ya) - model.predict(Xa)))
            preds.append(np.exp(model.predict(Xb)) * smear)
        return np.mean(preds, axis=0)
    if name in ("catboost", "catboost_te"):
        cols = FEATURES + CAT_COLS
        Xa, Xb = Fa[cols].copy(), Fb[cols].copy()
        if name == "catboost_te":
            Xa, Xb = pd.concat([Xa, te_a[TE_USE]], axis=1), pd.concat([Xb, te_b[TE_USE]], axis=1)
        preds = []
        for s in range(N_SEEDS):
            model = CatBoostRegressor(**CAT_PARAMS, random_seed=100 * seed + s, cat_features=CAT_COLS)
            preds.append(model.fit(Xa, ya).predict(Xb))
        return np.mean(preds, axis=0)
    raise ValueError(name)


def designs(Fa, Fb, te_a, te_b, categories):
    Xa = pd.concat([Fa[FEATURES + LOW_CARD_COLS], te_a[TE_USE]], axis=1)
    Xb = pd.concat([Fb[FEATURES + LOW_CARD_COLS], te_b[TE_USE]], axis=1)
    d = Design(FEATURES + TE_USE, LOW_CARD_COLS, categories).fit(Xa)
    k = Design(KNN_COLS, [], {}).fit(Xa)
    return d.transform(Xa), d.transform(Xb), k.transform(Xa), k.transform(Xb)


def run_model(m, Fa, ya, Fb, te_a, te_b, seed, categories, mats, failed, spent):
    if m in failed:
        return np.full(len(Fb), np.nan)
    t = time.time()
    try:
        if m in CORE:
            pred = fit_predict(m, Fa, ya, Fb, te_a, te_b, seed, categories)
        else:
            if m in GPU_FAMILIES and not cuda_available():
                raise RuntimeError("needs Accelerator = GPU (no CUDA device)")
            pred = predict_family(m, mats[0], ya, mats[1], seed, mats[2], mats[3])
        pred = np.asarray(pred, dtype=float)
        if not np.isfinite(pred).all():
            raise RuntimeError(f"{int((~np.isfinite(pred)).sum())} non-finite predictions")
    except Exception as e:
        failed[m] = f"{type(e).__name__}: {str(e)[:300]}"
        print(f"  !! {m} failed and is dropped: {failed[m]}")
        pred = np.full(len(Fb), np.nan)
    finally:
        if m in GPU_FAMILIES:
            free_gpu()
    spent[m] = spent.get(m, 0.0) + time.time() - t
    return pred


def nnls_weights(oof, y, models):
    A = np.column_stack([oof[m].ravel() for m in models])
    w, _ = nnls(A, np.tile(y, oof[models[0]].shape[0]))
    return dict(zip(models, w))


def nested_nnls(oof, y, models, folds):
    out = np.zeros_like(oof[models[0]])
    for r in range(folds.shape[0]):
        for f in np.unique(folds[r]):
            tr_rows, va = folds[r] != f, folds[r] == f
            A = np.column_stack([oof[m][r, tr_rows] for m in models])
            w, _ = nnls(A, y[tr_rows])
            out[r, va] = np.column_stack([oof[m][r, va] for m in models]) @ w
    return out


def report(oof, y, seen, weights):
    print(f"\n{'model':<12} {'RMSE mean':>10} {'sd':>7} {'MSE(mean RMSE)':>15} {'seen':>8} {'unseen':>8}")
    rows = {}
    for m, pred in oof.items():
        per = [rmse(pred[r], y) for r in range(pred.shape[0])]
        s = rmse(pred[seen], np.broadcast_to(y, pred.shape)[seen])
        u = rmse(pred[~seen], np.broadcast_to(y, pred.shape)[~seen])
        rows[m] = np.mean(per)
        print(f"{m:<12} {np.mean(per):>10.0f} {np.std(per):>7.0f} {np.mean(per) ** 2:>15.4g} {s:>8.0f} {u:>8.0f}")
    print(f"seen-community share of held-out rows: {seen.mean():.3f}")
    for name, wts in weights.items():
        print(f"NNLS weights [{name}]: " + ", ".join(f"{m}={w:.3f}" for m, w in sorted(wts.items(), key=lambda kv: -kv[1]) if w > 0.0005))
    return rows


def write_submission(test, pred, path):
    keep = test["ID"].notna().to_numpy()
    sub = pd.DataFrame({"ID": test.loc[keep, "ID"].astype(int).to_numpy(), "单价": pred[keep]})
    sub = sub.sort_values("ID").reset_index(drop=True)
    assert len(sub) == 581 and (sub["ID"].to_numpy() == np.arange(581)).all(), "host IDs are 0..580 — re-upload data.zip if test.csv still has 1..581"
    sub.to_csv(path, index=False)


def main():
    t0 = time.time()
    train, test = load_data()
    n_parking = int((train["layout"] == "parking_space").sum())
    train = train[train["layout"] != "parking_space"].reset_index(drop=True)
    print(f"{VERSION}: train {len(train)} rows (dropped {n_parking} parking spaces), test {len(test)} rows")
    y = train["unit_price"].to_numpy(dtype=float)
    pool_pred = pool_predictions(load_pool())
    Ftr = add_pool_features(build_features(train), train, pool_pred, "train")
    Fte = add_pool_features(build_features(test), test, pool_pred, "test")
    ly = np.log(y)
    for c in ["log_pool_px", "log_pool_px_loc", "log_px_combined"]:
        ok = Ftr[c].notna().to_numpy()
        print(f"corr(log y, {c}) = {np.corrcoef(ly[ok], Ftr[c].to_numpy()[ok])[0, 1]:.4f} on {ok.mean():.3f} of train; "
              f"test coverage {Fte[c].notna().mean():.3f}")
    categories = {c: sorted(set(Ftr[c]) | set(Fte[c])) for c in LOW_CARD_COLS}

    n = len(train)
    oof = {m: np.zeros((N_REPEATS, n)) for m in MODELS}
    seen = np.zeros((N_REPEATS, n), dtype=bool)
    folds = np.zeros((N_REPEATS, n), dtype=int)
    failed, spent = {}, {}
    print(f"CUDA available: {cuda_available()}")
    for r in range(N_REPEATS):
        for f, (a, b) in enumerate(KFold(N_FOLDS, shuffle=True, random_state=r).split(Ftr)):
            Fa, Fb = Ftr.iloc[a], Ftr.iloc[b]
            te_a, te_b, enc = encode_train_test(Fa, y[a], Fb, INNER_FOLDS, seed=100 * r + f)
            seen[r, b] = Fb["community_key"].isin(set(Fa["community_key"])).to_numpy()
            folds[r, b] = f
            mats = designs(Fa, Fb, te_a, te_b, categories)
            for m in MODELS:
                oof[m][r, b] = run_model(m, Fa, y[a], Fb, te_a, te_b, r, categories, mats, failed, spent)
            if r == 0 and f == 0:
                print("EB shrinkage k (district, street, community):", [round(k, 2) for k in enc.k_])
                print(f"design matrix: {mats[0].shape[1]} columns; knn: {mats[2].shape[1]}")
            print(f"  repeat {r} fold {f} done [{time.time() - t0:.0f}s]")
        live = [m for m in MODELS if m not in failed]
        print(f"repeat {r}: " + " ".join(f"{m}={rmse(oof[m][r], y):.0f}" for m in live) + f"  [{time.time() - t0:.0f}s]")
    print("seconds per family (all folds): " + ", ".join(f"{m} {spent[m]:.0f}" for m in MODELS if m in spent))

    active = [m for m in MODELS if m not in failed]
    core = [m for m in CORE if m not in failed]
    weights = {"all": nnls_weights(oof, y, active), "core": nnls_weights(oof, y, core)}
    scored = {m: oof[m] for m in active}
    scored["blend"] = np.tensordot([weights["all"][m] for m in active], np.stack([oof[m] for m in active]), axes=1)
    scored["blend_core"] = np.tensordot([weights["core"][m] for m in core], np.stack([oof[m] for m in core]), axes=1)
    scored["nested_all"] = nested_nnls(oof, y, active, folds)
    scored["nested_core"] = nested_nnls(oof, y, core, folds)
    report(scored, y, seen, weights)
    print("blend/blend_core: weights fitted on the OOF they score (optimistic); nested_*: weights fitted on the other folds")

    out = get_out_dir()
    pd.DataFrame({"row_id": train["row_id"], "y": y, **{m: scored[m].mean(axis=0) for m in scored}}).to_csv(
        out / f"oof_{VERSION.lower().replace('.', '_')}.csv", index=False)

    te_tr, te_te, enc = encode_train_test(Ftr, y, Fte, INNER_FOLDS, seed=999)
    print("full-data EB k:", [round(k, 2) for k in enc.k_])
    mats = designs(Ftr, Fte, te_tr, te_te, categories)
    test_pred = {m: run_model(m, Ftr, y, Fte, te_tr, te_te, 0, categories, mats, failed, spent) for m in active}
    if any(m in failed for m in active):
        lost = [m for m in active if m in failed]
        print(f"!! final fit failed for {lost}; refitting NNLS without them")
        active = [m for m in active if m not in failed]
        core = [m for m in core if m not in failed]
        weights = {"all": nnls_weights(oof, y, active), "core": nnls_weights(oof, y, core)}
    lo, hi = y.min(), y.max()
    blend = np.clip(sum(weights["all"][m] * test_pred[m] for m in active), lo, hi)
    blend_core = np.clip(sum(weights["core"][m] * test_pred[m] for m in core), lo, hi)
    subdir = out / "subs"
    subdir.mkdir(exist_ok=True)
    for m in active:
        write_submission(test, np.clip(test_pred[m], lo, hi), subdir / f"submission_{m}.csv")
    write_submission(test, blend_core, subdir / "submission_core.csv")
    write_submission(test, blend, out / "submission.csv")
    keep = test["ID"].notna().to_numpy()
    print(f"submission.csv = NNLS blend of {len(active)} families; subs/submission_core.csv = V3's six models only")
    print(f"rms(all - core) on scored test {np.sqrt(np.mean((blend[keep] - blend_core[keep]) ** 2)):.0f}; "
          f"scored-test mean {blend[keep].mean():.0f} vs train mean {y.mean():.0f}")
    if failed:
        print("failed families: " + "; ".join(f"{m}: {e}" for m, e in failed.items()))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
