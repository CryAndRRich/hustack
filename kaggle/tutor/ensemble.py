import gc
import importlib
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import (ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

from sklearn.decomposition import PCA

from preprocess import (CAT_COLS, EXT_CAT_COLS, ID, NAME, TARGET, build_features, ext_features, extra_features,
                        fill_rating_by_name, get_out_dir, load_data, load_external, price_features, restore_price,
                        teacher_wage_features)
from text_embed import emb_checksum, embed_frames, embed_st, tfidf_frames
from train import (CV_SEED, EMB_ALPHAS, TFIDF_ALPHAS, cross_subject_copy, emb_stack, fit_predict, fold_ids, make_folds, mse,
                   name_lookup, person_key)

VERSION = "V5"
MEMBERS = {
    "cat_v4": ("cat", "v4", "all"), "tabpfn_v4": ("tabpfn", "v4", "all"),
    "cat": ("cat", "f5", "all"), "cat_single": ("cat", "f5", "single"),
    "lgbm": ("lgbm", "f5", "all"), "xgb": ("xgb", "f5", "all"), "hgb": ("hgb", "f5", "all"),
    "forest": ("forest", "f5", "all"), "gbr": ("gbr", "f5", "all"), "linear": ("linear", "f5", "all"),
    "huber": ("huber", "f5", "all"), "krr": ("krr", "f5", "all"), "knn": ("knn", "f5", "all"),
    "mlp": ("mlp", "f5", "all"), "tabpfn": ("tabpfn", "f5", "all"), "tabicl": ("tabicl", "f5", "all"),
    "realmlp": ("realmlp", "f5", "all"), "tabm": ("tabm", "f5", "all"), "tabdpt": ("tabdpt", "f5", "all"),
}
GPU_FAMILIES = {"tabpfn", "tabicl", "realmlp", "tabm", "tabdpt"}
PIP = {"tabpfn": "tabpfn>=2,<3", "tabicl": "tabicl>=2.2,<3", "pytabkit": "pytabkit>=1.7,<2", "tabdpt": "tabdpt>=1.3,<2"}
REPRODUCE_CAT, REPRODUCE_TOL = 61.01, 0.8
KNN_STACK_K = 30
N_PCA = 16
RECAL_MIN_GAIN = 0.2
FINAL_SEEDS = [0, 1, 2]
TE_SMOOTH = 20.0
BLEND_RIDGE = 0.05
SMOKE_ROWS = 300
KNN_FEATURES = ["emb_ridge", "loc_wage", "edu_max_ege", "experience", "desc_words", "st_student", "kw_n",
                "n_pair", "rating", "log_reviews", "cat_g10_11", "loc_msk", "expdesc_len", "n_copies"]
HGB_MEMBERS = [
    dict(max_iter=600, learning_rate=0.04, max_leaf_nodes=63, min_samples_leaf=15, l2_regularization=1.0),
    dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=1.0),
    dict(max_iter=900, learning_rate=0.02, max_leaf_nodes=31, min_samples_leaf=30, l2_regularization=3.0),
]


def pip_import(module, spec):
    try:
        return importlib.import_module(module)
    except ImportError:
        pass
    print(f"  installing {spec} (needs Internet ON) ...")
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", spec], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"pip install {spec} failed: {r.stderr.strip()[-300:]}")
    importlib.invalidate_caches()
    return importlib.import_module(module)


def cuda_available():
    try:
        import torch
        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def te_map(vals, y, prior):
    s = pd.DataFrame({"c": vals, "y": y}).groupby("c")["y"].agg(["sum", "count"])
    return (s["sum"] + prior * TE_SMOOTH) / (s["count"] + TE_SMOOTH)


def target_encode(Xa, ya, ga, Xbs, cols, seed):
    prior = float(np.mean(ya))
    A, Bs = {}, [{} for _ in Xbs]
    for c in cols:
        enc = np.full(len(Xa), prior)
        va = Xa[c].to_numpy()
        for a, b in make_folds(ga, seed):
            mp = te_map(va[a], ya[a], prior)
            enc[b] = pd.Series(va[b]).map(mp).fillna(prior).to_numpy()
        A[f"{c}_te"] = enc
        mp = te_map(va, ya, prior)
        for B, Xb in zip(Bs, Xbs):
            B[f"{c}_te"] = Xb[c].map(mp).fillna(prior).to_numpy()
    return pd.DataFrame(A), [pd.DataFrame(B) for B in Bs]


def design(Xa, ya, ga, Xbs, seed, fill, standardise):
    cats = [c for c in CAT_COLS + EXT_CAT_COLS if c in Xa.columns]
    num = [c for c in Xa.columns if c not in cats]
    tA, tBs = target_encode(Xa, ya, ga, Xbs, cats, seed)
    A = pd.concat([Xa[num].astype(float).reset_index(drop=True), tA], axis=1).replace([np.inf, -np.inf], np.nan)
    Bs = [pd.concat([Xb[num].astype(float).reset_index(drop=True), t], axis=1).replace([np.inf, -np.inf], np.nan)
          for Xb, t in zip(Xbs, tBs)]
    keep = A.columns[A.notna().any()]
    A, Bs = A[keep], [B[keep] for B in Bs]
    if fill:
        nan_cols = [c for c in A.columns if A[c].isna().any()]
        for D in [A] + Bs:
            for c in nan_cols:
                D[f"{c}_na"] = D[c].isna().astype(float)
        med = A.median().fillna(0.0)
        A, Bs = A.fillna(med), [B[A.columns].fillna(med) for B in Bs]
    if standardise:
        mu, sd = A.mean(), A.std(ddof=0).replace(0.0, 1.0)
        A, Bs = (A - mu) / sd, [(B[A.columns] - mu) / sd for B in Bs]
    return A, [B[A.columns] for B in Bs]


def designs(Xa, ya, ga, Xbs, seed):
    return {"raw": design(Xa, ya, ga, Xbs, seed, fill=False, standardise=False),
            "fill": design(Xa, ya, ga, Xbs, seed, fill=True, standardise=False),
            "std": design(Xa, ya, ga, Xbs, seed, fill=True, standardise=True)}


def fit_family(fam, Xa, ya, Xbs, D, seed):
    if fam == "cat":
        out = [np.zeros(len(Xb)) for Xb in Xbs]
        for Xb_i, Xb in enumerate(Xbs):
            out[Xb_i] = fit_predict(Xa, ya, Xb, seed)[0]
        return out
    if fam in ("lgbm", "xgb", "hgb"):
        A, Bs = D["raw"]
        if fam == "lgbm":
            import lightgbm as lgb
            models = [lgb.LGBMRegressor(n_estimators=1500, learning_rate=0.02, num_leaves=31, min_child_samples=20,
                                        subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=1.0,
                                        random_state=seed, verbose=-1)]
        elif fam == "xgb":
            import xgboost as xgb
            models = [xgb.XGBRegressor(n_estimators=1500, learning_rate=0.02, max_depth=5, min_child_weight=3,
                                       subsample=0.8, colsample_bytree=0.6, reg_lambda=1.0, tree_method="hist",
                                       random_state=seed, verbosity=0, n_jobs=-1)]
        else:
            models = [HistGradientBoostingRegressor(random_state=seed, early_stopping=False, **m) for m in HGB_MEMBERS]
        return average(models, A, ya, Bs)
    if fam == "forest":
        A, Bs = D["fill"]
        models = [RandomForestRegressor(n_estimators=500, min_samples_leaf=5, max_features=0.5, random_state=seed, n_jobs=-1),
                  ExtraTreesRegressor(n_estimators=500, min_samples_leaf=5, max_features=0.7, random_state=seed, n_jobs=-1)]
        return average(models, A, ya, Bs)
    if fam == "gbr":
        A, Bs = D["fill"]
        return average([GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=4, min_samples_leaf=15,
                                                  subsample=0.8, max_features=0.5, random_state=seed)], A, ya, Bs)
    A, Bs = D["std"]
    if fam == "linear":
        return average([Ridge(alpha=30.0)], A, ya, Bs)
    if fam == "huber":
        return average([HuberRegressor(epsilon=1.35, alpha=1e-3, max_iter=2000)], A, ya, Bs)
    if fam == "krr":
        ny = Nystroem(kernel="rbf", gamma=1.0 / A.shape[1], n_components=min(1000, len(A)), random_state=seed)
        Z = ny.fit_transform(A)
        model = Ridge(alpha=1.0).fit(Z, ya)
        return [model.predict(ny.transform(B)) for B in Bs]
    if fam == "knn":
        cols = [c for c in KNN_FEATURES if c in A.columns]
        model = KNeighborsRegressor(n_neighbors=min(25, len(A)), weights="distance", n_jobs=-1).fit(A[cols], ya)
        return [model.predict(B[cols]) for B in Bs]
    if fam == "mlp":
        common = dict(max_iter=400, early_stopping=True, n_iter_no_change=15, validation_fraction=0.12,
                      learning_rate_init=1e-3, batch_size=64)
        models = [MLPRegressor(hidden_layer_sizes=(128, 64), alpha=1e-2, random_state=seed, **common),
                  MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-3, random_state=seed + 101, **common)]
        return average(models, A, ya, Bs)
    if fam == "tabpfn":
        cls = pip_import("tabpfn", PIP["tabpfn"]).TabPFNRegressor
        model = cls(device="cuda", n_estimators=8, ignore_pretraining_limits=True, random_state=seed)
        return average([model], A, ya, Bs)
    if fam == "tabicl":
        cls = pip_import("tabicl", PIP["tabicl"]).TabICLRegressor
        return average([cls(random_state=seed, verbose=False, n_estimators=8)], A, ya, Bs)
    if fam == "tabdpt":
        mod = pip_import("tabdpt", PIP["tabdpt"])
        model = mod.TabDPTRegressor(context_reduction="subsample", compile=False, use_flash=False, verbose=False)
        model.fit(A.to_numpy(dtype=np.float32), ya.astype(np.float32))
        return [np.asarray(model.predict(B.to_numpy(dtype=np.float32), n_ensembles=8, context_size=None,
                                         batch_size=1024, seed=seed)).ravel() for B in Bs]
    if fam in ("realmlp", "tabm"):
        mod = pip_import("pytabkit", PIP["pytabkit"])
        cls = mod.RealMLP_TD_Regressor if fam == "realmlp" else mod.TabM_D_Regressor
        model = cls(random_state=seed, verbosity=0)
        model.fit(A, ya.astype(np.float32))
        return [np.asarray(model.predict(B)).ravel() for B in Bs]
    raise ValueError(fam)


def average(models, A, ya, Bs):
    acc = [np.zeros(len(B)) for B in Bs]
    for m in models:
        m.fit(A, ya)
        for i, B in enumerate(Bs):
            acc[i] += np.asarray(m.predict(B)).ravel() / len(models)
    return acc


def knn_oof(E, y, groups, seed, k=KNN_STACK_K):
    pred = np.zeros(len(y))
    for a, b in make_folds(groups, seed):
        pred[b] = KNeighborsRegressor(n_neighbors=min(k, len(a)), metric="cosine", weights="distance",
                                      algorithm="brute").fit(E[a], y[a]).predict(E[b])
    return pred


def knn_stack(Ea, ya, ga, Eb, seed):
    has_a, has_b = np.abs(Ea).sum(axis=1) > 0, np.abs(Eb).sum(axis=1) > 0
    fa, fb = np.full(len(Ea), np.nan), np.full(len(Eb), np.nan)
    fa[has_a] = knn_oof(Ea[has_a], ya[has_a], ga[has_a], seed)
    model = KNeighborsRegressor(n_neighbors=min(KNN_STACK_K, int(has_a.sum())), metric="cosine", weights="distance",
                                algorithm="brute").fit(Ea[has_a], ya[has_a])
    if has_b.any():
        fb[has_b] = model.predict(Eb[has_b])
    return fa, fb


def project_simplex(v):
    u = np.sort(v)[::-1]
    css = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(v) + 1) > (css - 1))[0][-1]
    return np.maximum(v - (css[rho] - 1) / (rho + 1.0), 0.0)


def simplex_weights(P, y, ridge=BLEND_RIDGE):
    n = P.shape[1]
    prior = np.full(n, 1.0 / n)
    w = prior.copy()
    lr = 1.0 / (np.linalg.norm(P, 2) ** 2 / len(P) + ridge)
    for _ in range(3000):
        w = project_simplex(w - lr * (P.T @ (P @ w - y) / len(P) + ridge * (w - prior)))
    return w


def main():
    t0 = time.time()
    train, test = load_data()
    p, low = restore_price(train[TARGET].to_numpy(dtype=float))
    fill_rating_by_name(train, test)
    counts = pd.concat([train[NAME], test[NAME]]).value_counts()
    single = (train[NAME].map(counts) == 1).to_numpy()
    seen = test[NAME].isin(set(train[NAME])).to_numpy()
    groups = train[NAME].to_numpy()
    print(f"{VERSION}: train {len(train)} (singletons {single.sum()}), test {len(test)} (lookup {seen.sum()}); "
          f"restored x100: {low.sum()}")

    ext = load_external()
    e_tr, e_te = ext_features(train, ext), ext_features(test, ext)
    X4 = pd.concat([build_features(train), e_tr, extra_features(train, counts)], axis=1)
    X4t = pd.concat([build_features(test), e_te, extra_features(test, counts)], axis=1)
    X5 = pd.concat([X4, price_features(train), teacher_wage_features(e_tr, ext)], axis=1)
    X5t = pd.concat([X4t, price_features(test), teacher_wage_features(e_te, ext)], axis=1)
    E_tr, E_te = embed_frames(train, test)
    T_tr, T_te = tfidf_frames(train, test)
    B_tr, B_te, Dv_tr, Dv_te = embed_st(train, test)
    pca = PCA(n_components=N_PCA, random_state=0).fit(np.vstack([Dv_tr, Dv_te]))
    pc_tr, pc_te = pca.transform(Dv_tr), pca.transform(Dv_te)
    for i in range(N_PCA):
        X5[f"bge_pc{i}"], X5t[f"bge_pc{i}"] = pc_tr[:, i], pc_te[:, i]
    assert list(X4.columns) == list(X4t.columns) and list(X5.columns) == list(X5t.columns)
    print(f"feature sets: v4 {X4.shape[1]} + emb_ridge; f5 {X5.shape[1]} + 4 stacks; rubert checksum "
          f"{emb_checksum(E_tr, E_te)}; bge {B_tr.shape[1]} dims (PCA {N_PCA}: {pca.explained_variance_ratio_.sum():.2f} var)  "
          f"[{time.time() - t0:.0f}s]")
    S4 = {"emb_ridge": ("ridge", E_tr, EMB_ALPHAS)}
    S5 = {"emb_ridge": ("ridge", E_tr, EMB_ALPHAS), "tfidf_ridge": ("ridge", T_tr, TFIDF_ALPHAS),
          "bge_ridge": ("ridge", B_tr, EMB_ALPHAS), "bge_knn": ("knn", Dv_tr, None)}
    sets = {"v4": (X4, X4t, S4), "f5": (X5, X5t, S5)}

    gpu = cuda_available()
    members = {m: spec for m, spec in MEMBERS.items() if gpu or spec[0] not in GPU_FAMILIES}
    print(f"CUDA {'available' if gpu else 'absent'}: {len(members)} members = {', '.join(members)}")

    idx = np.random.default_rng(0).choice(len(train), SMOKE_ROWS, replace=False)
    a, b = idx[:SMOKE_ROWS - 50], idx[SMOKE_ROWS - 50:]
    ok = []
    for m, (fam, fs, pop) in members.items():
        X, _, S = sets[fs]
        Xa = X.iloc[a].assign(**{k: p[a] for k in S})
        Xb = X.iloc[b].assign(**{k: p[b] for k in S})
        try:
            D = designs(Xa, p[a], groups[a], [Xb], seed=0) if fam != "cat" else None
            pred = fit_family(fam, Xa, p[a], [Xb], D, seed=0)[0]
            assert np.isfinite(pred).all() and len(pred) == len(b)
            ok.append(m)
        except Exception as e:
            print(f"  smoke: {m} FAILED, dropped ({type(e).__name__}: {str(e)[:200]})")
        gc.collect()
    members = {m: members[m] for m in ok}
    print(f"smoke pass ok: {', '.join(members)}  [{time.time() - t0:.0f}s]")

    def frames(fs, rows_a, rows_b, st_a, st_b, X_b_src=None):
        X, Xt, S = sets[fs]
        Xa = X.iloc[rows_a].assign(**{k: st_a[k] for k in S})
        src = X if X_b_src is None else X_b_src
        Xb = src.iloc[rows_b].assign(**{k: st_b[k] for k in S})
        return Xa, Xb

    oof = {m: np.full(len(train), np.nan) for m in members}
    secs = {m: 0.0 for m in members}
    folds = make_folds(groups, CV_SEED)
    for k, (a_idx, b_idx) in enumerate(folds):
        st_a, st_b = {}, {}
        for name, (kind, M, al) in {**S4, **S5}.items():
            if kind == "ridge":
                st_a[name], st_b[name], _ = emb_stack(M[a_idx], p[a_idx], groups[a_idx], M[b_idx], seed=CV_SEED + 100, alphas=al)
            else:
                st_a[name], st_b[name] = knn_stack(M[a_idx], p[a_idx], groups[a_idx], M[b_idx], seed=CV_SEED + 100)
        cache = {}
        for m, (fam, fs, pop) in members.items():
            t1 = time.time()
            keep = single[a_idx] if pop == "single" else np.ones(len(a_idx), bool)
            rows = a_idx[keep]
            Xa, Xb = frames(fs, rows, b_idx, {kk: v[keep] for kk, v in st_a.items()}, st_b)
            try:
                if fam == "cat":
                    D = None
                else:
                    key = (fs, pop)
                    if key not in cache:
                        cache[key] = designs(Xa, p[rows], groups[rows], [Xb], seed=CV_SEED + 300 + k)
                    D = cache[key]
                oof[m][b_idx] = fit_family(fam, Xa, p[rows], [Xb], D, seed=0)[0]
            except Exception as e:
                print(f"  fold {k}: {m} failed ({type(e).__name__}: {str(e)[:200]})")
            secs[m] += time.time() - t1
            gc.collect()
        print(f"fold {k}: " + "  ".join(f"{m}={mse(p[b_idx][single[b_idx]], oof[m][b_idx][single[b_idx]]):.1f}"
                                        for m in members if np.isfinite(oof[m][b_idx]).all()) + f"  [{time.time() - t0:.0f}s]")
        stack_diag = {kk: mse(p[b_idx][single[b_idx] & np.isfinite(v)], v[single[b_idx] & np.isfinite(v)])
                      for kk, v in st_b.items()}
        print("   stacks alone (singletons): " + "  ".join(f"{kk}={v:.1f}" for kk, v in stack_diag.items()))
    members = {m: s_ for m, s_ in members.items() if np.isfinite(oof[m]).all()}
    names = list(members)

    if "cat_v4" in oof:
        c4 = mse(p, oof["cat_v4"])
        print(f"\nplumbing: cat_v4 all-row OOF {c4:.2f} vs V4 cat {REPRODUCE_CAT:.2f} (tol {REPRODUCE_TOL}) -> "
              f"{'REPRODUCED' if abs(c4 - REPRODUCE_CAT) <= REPRODUCE_TOL else 'MISMATCH'}")
    ys = p[single]
    P = np.column_stack([oof[m] for m in names])
    frac = (~seen).mean()
    print(f"singleton OOF MSE (constant {np.var(ys):.2f}); expected LB = {frac:.3f} x singleton")
    for m in names:
        print(f"  {m:<11} {mse(ys, oof[m][single]):6.2f}   {members[m][1]}/{members[m][2]:<6}  {secs[m]:.0f}s")
    for new, old in [("cat", "cat_v4"), ("tabpfn", "tabpfn_v4")]:
        if new in oof and old in oof:
            dd = ((p - oof[new]) ** 2 - (p - oof[old]) ** 2)[single]
            print(f"  feature effect {new} - {old}: {dd.mean():+.2f} (SE {dd.std() / np.sqrt(len(dd)):.2f})")

    f_id = fold_ids(groups, CV_SEED)
    w_fit = simplex_weights(P[single], ys)
    nested = np.full(len(train), np.nan)
    for k in range(len(folds)):
        tr_m, te_m = single & (f_id != k), single & (f_id == k)
        nested[te_m] = P[te_m] @ simplex_weights(P[tr_m], p[tr_m])
    v4_names = [m for m in names if members[m][1] == "v4"]
    print(f"  equal       {mse(ys, P[single].mean(axis=1)):6.2f}")
    print(f"  simplex     {mse(ys, P[single] @ w_fit):6.2f} (in-sample)   nested {mse(ys, nested[single]):6.2f}")
    print("  weights: " + ", ".join(f"{m}={w:.3f}" for m, w in zip(names, w_fit) if w > 1e-3))

    recal = np.full(len(train), np.nan)
    for k in range(len(folds)):
        tr_m, te_m = single & (f_id != k), single & (f_id == k)
        bb, aa = np.polyfit(nested[tr_m], p[tr_m], 1)
        recal[te_m] = aa + bb * nested[te_m]
    bb_all, aa_all = np.polyfit(nested[single], ys, 1)
    gain = mse(ys, nested[single]) - mse(ys, recal[single])
    use_recal = gain > RECAL_MIN_GAIN
    print(f"  recalibration p ~ {aa_all:.2f} + {bb_all:.3f} * blend: nested gain {gain:+.2f} -> {'APPLY' if use_recal else 'skip'}")

    resid = pd.DataFrame({"r": (p - nested)[single], "pred": nested[single], "p": ys,
                          "city": X4["loc_city"].to_numpy()[single], "student": X4["st_student"].to_numpy()[single],
                          "desc_empty": X4["desc_empty"].to_numpy()[single]})
    print("  residual check (singletons, nested blend): by predicted decile mean(p - pred) / mean p:")
    dec = pd.qcut(resid["pred"], 10, labels=False, duplicates="drop")
    print("   " + "  ".join(f"{d}:{g['r'].mean():+.1f}/{g['p'].mean():.1f}" for d, g in resid.groupby(dec)))
    top = resid["city"].value_counts().index[:6]
    print("   by city: " + "  ".join(f"{c}:{resid.loc[resid['city'] == c, 'r'].mean():+.2f}" for c in top)
          + f"  student:{resid.loc[resid['student'] == 1, 'r'].mean():+.2f}  empty desc:{resid.loc[resid['desc_empty'] == 1, 'r'].mean():+.2f}")

    person_tr, person_te = person_key(train[NAME]), person_key(test[NAME])
    d_copy, d_model = [], []
    for k in range(len(folds)):
        q = np.where((f_id == k) & single)[0]
        cp = cross_subject_copy(person_tr, groups, p, person_tr[q], groups[q], f_id != k)
        mm = ~np.isnan(cp)
        d_copy += list((p[q][mm] - cp[mm]) ** 2)
        d_model += list((p[q][mm] - nested[q][mm]) ** 2)
    use_d = len(d_copy) > 0 and np.mean(d_copy) < np.mean(d_model)
    print(f"block D on singleton OOF: {len(d_copy)} rows, copy {np.mean(d_copy) if d_copy else float('nan'):.2f} vs "
          f"blend {np.mean(d_model) if d_model else float('nan'):.2f} -> {'APPLY' if use_d else 'skip'}")

    all_idx = np.arange(len(train))
    st_a, st_t = {}, {}
    for name, (kind, M, al) in {**S4, **S5}.items():
        Mt = {"emb_ridge": E_te, "tfidf_ridge": T_te, "bge_ridge": B_te, "bge_knn": Dv_te}[name]
        if kind == "ridge":
            st_a[name], st_t[name], _ = emb_stack(M, p, groups, Mt, seed=CV_SEED + 999, alphas=al)
        else:
            st_a[name], st_t[name] = knn_stack(M, p, groups, Mt, seed=CV_SEED + 999)
    test_pred = {}
    for m in names:
        fam, fs, pop = members[m]
        t1 = time.time()
        keep = single if pop == "single" else np.ones(len(train), bool)
        rows = all_idx[keep]
        X, Xt, S = sets[fs]
        Xa = X.iloc[rows].assign(**{kk: st_a[kk][keep] for kk in S})
        Xb = Xt.assign(**{kk: st_t[kk] for kk in S})
        seeds = FINAL_SEEDS if fam in ("cat", "lgbm", "xgb", "mlp") else FINAL_SEEDS[:1]
        acc = np.zeros(len(test))
        for sd in seeds:
            D = designs(Xa, p[rows], groups[rows], [Xb], seed=CV_SEED + 900 + sd) if fam != "cat" else None
            acc += fit_family(fam, Xa, p[rows], [Xb], D, seed=sd)[0] / len(seeds)
        test_pred[m] = acc
        print(f"  full fit {m}: {len(seeds)} seed(s), {time.time() - t1:.0f}s, model-row mean {acc[~seen].mean():.2f}")
        gc.collect()
    T = np.column_stack([test_pred[m] for m in names])

    look = name_lookup(train, p, test)
    d_test = cross_subject_copy(person_tr, groups, p, person_te, test[NAME].to_numpy(), np.ones(len(train), bool))
    d_rows = ~seen & ~np.isnan(d_test)
    blend = T @ w_fit
    if use_recal:
        blend = aa_all + bb_all * blend
    files = {"submission.csv": blend}
    if v4_names:
        files["submission_v4members.csv"] = T[:, [names.index(m) for m in v4_names]].mean(axis=1)
    files["submission_cat.csv"] = test_pred["cat"] if "cat" in test_pred else T[:, 0]
    out = get_out_dir()
    finals = {}
    for fname, pred in files.items():
        final = np.where(seen, look, np.clip(pred, p.min(), p.max()))
        if use_d:
            final = np.where(d_rows, d_test, final)
        finals[fname] = final
        pd.DataFrame({ID: test[ID].to_numpy(), TARGET: final}).to_csv(out / fname, index=False)
    for fname, final in finals.items():
        diff = (final - finals["submission.csv"])[~seen]
        print(f"{fname}: model-row mean {final[~seen].mean():.2f}; vs submission.csv mean sq diff {np.mean(diff ** 2):.2f}")
    pd.DataFrame({NAME: train[NAME], "p": p, "single": single, **{f"oof_{m}": oof[m] for m in names},
                  "blend_nested": nested}).to_csv(out / "oof_v5.csv", index=False)
    pd.DataFrame({ID: test[ID], **{m: test_pred[m] for m in names}}).to_csv(out / "test_members_v5.csv", index=False)
    print(f"submission.csv = simplex blend{' + recal' if use_recal else ''}{' + D' if use_d else ''} (+ lookup); "
          f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
