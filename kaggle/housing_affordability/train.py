import gc
import hashlib
import json
import sys
import time
import zipfile

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.cross_decomposition import PLSRegression
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import LinearSVR

try:
    import lightgbm as lgb
except ImportError:
    lgb = None
try:
    import xgboost as xgb
except ImportError:
    xgb = None
try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None
try:
    from tabpfn import TabPFNRegressor
except ImportError:
    TabPFNRegressor = None

from preprocess import (
    BURDEN_THRESHOLD,
    CATEGORICAL,
    GEO_K,
    ID,
    MONTH,
    NEIGHBOUR_K,
    NUMERIC,
    OOF_FOLDS,
    OOF_SEED,
    SMOOTHING,
    TARGET,
    TRAIN_MONTHS,
    add_feature_neighbour_features,
    add_geo_neighbour_features,
    add_neighbour_features,
    add_relative_features,
    add_ratio_features,
    get_output_dir,
    load_external,
    load_raw,
    merge_external,
    oof_fingerprint,
    oof_folds,
)
from stack import (
    align,
    bundle_zip,
    level2_fit_predict,
    level2_oof,
    blend_rmse,
    build_family_oof,
    load_families,
    optimise_weights,
    save_family,
    store_dir,
)

VERSION = "V10.3"

RUN_FAMILY = None

NBR_RESID_K = 10

ACS_VALUE = False

SHIP_FAMILIES = ["cat", "forest", "gbr", "hgb", "huber", "knn", "lgbm", "linear", "mlp", "pls",
                 "svr", "tabpfn", "xgb", "tabicl", "tabdpt", "realmlp", "tabm", "krr"]

TUNE_BUDGET_MIN = 150

TUNE_MAX_TRIALS = 60

TUNE_SEED = OOF_SEED + 1

TUNE_MIN_GAIN = 0.02

AUTO_SMOKE = True

SMOKE_MAX_ITER = 8

LOG_INCOME_CENTRE = 11.4

ENCODED_EXTRA = ("zip3", "county")


def _is_cat(f: str) -> bool:
    return f in CATEGORICAL or f in ENCODED_EXTRA

NATIVE_CAT_MAX = 255

BASE_FEATURES = NUMERIC + [
    "cost_ratio", "log_rent", "log_income", "income_per_person",
    "ratio_resid_income", "ratio_x_renter", "over_threshold",
    "State", "zip3", "nbr_mean",
]

ACS_VALUE_FEATURES = ["val_share_le_2x", "val_share_le_3x", "val_share_le_4x", "val_share_le_5x"]

EXTERNAL_FEATURES = [
    "price_to_income", "log_price_to_income", "rent_to_price", "log_zhvi",
    "irs_sh1", "irs_sh2", "irs_sh3", "irs_sh4", "irs_sh5", "irs_sh6",
    "irs_mean_agi", "irs_gini", "irs_top_share", "irs_n",
    "lat", "lon",
    "pti_bot", "pti_top", "bot_over_mid", "top_over_bot",

    "pti_br1", "pti_br2", "pti_br3", "pti_br4", "pti_sfr", "pti_condo",
    "condo_over_sfr", "br4_over_br2",
    "qcew_avg_wkly_wage", "qcew_emplvl", "qcew_estabs", "qcew_avg_annual_pay",
    "wage_to_income", "price_to_wage",
    "fhfa_hpi", "fhfa_chg_1y", "fhfa_chg_5y_mean", "fhfa_hpi_cum5y", "fhfa_stale_years",

    "nbr_geo_mean", "nbr_geo_std", "nbr_geo_dist", "pti_relative", "county",

    "feat_nbr_std",
]
if ACS_VALUE:
    EXTERNAL_FEATURES = EXTERNAL_FEATURES + ACS_VALUE_FEATURES


def impute_external(fit: pd.DataFrame, frames: list[pd.DataFrame],
                    cols: list[str]) -> list[pd.DataFrame]:

    present = [c for c in cols if c in fit.columns
               and pd.api.types.is_numeric_dtype(fit[c])]
    if not present:
        return list(frames)

    have = fit
    by_zip3 = have.groupby(have[ID].str[:3])[present].median()
    by_metro = have.groupby("Metro")[present].median()
    by_state = have.groupby("State")[present].median()
    overall = have[present].median()

    out = []
    for df in frames:
        res = df.copy()
        for c in present:
            miss = res[c].isna()
            if not miss.any():
                continue
            fill = res.loc[miss, ID].str[:3].map(by_zip3[c])
            fill = fill.fillna(res.loc[miss, "Metro"].map(by_metro[c]))
            fill = fill.fillna(res.loc[miss, "State"].map(by_state[c]))
            res.loc[miss, c] = fill.fillna(overall[c]).to_numpy()
        out.append(res)
    return out


def impute_rent(fit: pd.DataFrame, frames: list[pd.DataFrame]) -> list[pd.DataFrame]:

    have = fit[fit["RentIndex"].notna()]
    by_zip3 = have.groupby(have[ID].str[:3])["RentIndex"].median()
    by_metro = have.groupby("Metro")["RentIndex"].median()
    by_state = have.groupby("State")["RentIndex"].median()
    overall = float(have["RentIndex"].median()) if len(have) else 0.0

    out = []
    for df in frames:
        res = df.copy()
        miss = res["RentIndex"].isna()
        if miss.any():
            fill = res.loc[miss, ID].str[:3].map(by_zip3)
            fill = fill.fillna(res.loc[miss, "Metro"].map(by_metro))
            fill = fill.fillna(res.loc[miss, "State"].map(by_state))
            res.loc[miss, "RentIndex"] = fill.fillna(overall).to_numpy()
        out.append(res)
    return out


def add_interactions(df: pd.DataFrame) -> pd.DataFrame:

    out = df.copy()
    out["ratio_resid_income"] = out["cost_ratio"] * (out["log_income"] - LOG_INCOME_CENTRE)
    out["ratio_x_renter"] = out["cost_ratio"] * (100.0 - out["OwnerPercent"]) / 100.0
    out["over_threshold"] = (out["cost_ratio"] > BURDEN_THRESHOLD).astype(float)
    return out


def _target_encode(fit: pd.DataFrame, frames: list[pd.DataFrame], col: str) -> list[pd.Series]:

    prior = float(fit[TARGET].mean())
    stats = fit.groupby(col)[TARGET].agg(["mean", "count"])
    enc = (stats["mean"] * stats["count"] + prior * SMOOTHING) / (stats["count"] + SMOOTHING)
    return [df[col].map(enc).fillna(prior).astype(float) for df in frames]


def load_all(verbose: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    train, test, sample = load_raw()
    ext = load_external()
    train, test = merge_external(train, ext), merge_external(test, ext)
    missing = [f for f in ACS_VALUE_FEATURES if ACS_VALUE and f not in test.columns]
    if missing:

        raise RuntimeError(f"ACS_VALUE is on but {missing} were not built: upload zip_acs_value.csv "
                           f"into the external_data dataset")

    if verbose:
        got = [f for f in EXTERNAL_FEATURES if f in test.columns]
        if not got:
            print("  !! NO EXTERNAL FEATURES -- this run will score like V1 (~10.75), not ~5.4")
        else:
            trz = set(train[ID])
            new = ~test[ID].isin(trz)
            print(f"  external columns merged: {len(got)} of {len(EXTERNAL_FEATURES)} -- the rest "
                  f"are built per fold; see 'features reaching the model' when storing")
            for c in ("price_to_income", "irs_sh6", "lat"):
                if c in test.columns:
                    miss = int(test.loc[new, c].isna().sum())
                    print(f"    {c:18s} missing for {miss}/{int(new.sum())} new zips")
    return train, test, sample


def active_features(fit: pd.DataFrame) -> list[str]:

    return BASE_FEATURES + [f for f in EXTERNAL_FEATURES if f in fit.columns]

STORE_SCHEMA_VERSION = "v9"

FAMILY_PARAMS = {

    "hgb": {"max_leaf_nodes": 32, "min_samples_leaf": 39, "l2_regularization": 0.05962591584398859,
            "max_bins": 176},

    "forest": {"rf_max_features": 0.5828526894940085, "et_max_features": 0.8202408165890005,
               "min_samples_leaf": 5},

    "mlp": {"hidden_a": [256, 128], "hidden_b": [32, 16], "alpha_a": 0.014812183816486213,
            "alpha_b": 0.00010874859322697434, "lr": 0.0007590506182232217, "batch_size": 64},

    "knn": {"n_neighbors": 23, "weights": "distance", "p": 1},

    "pls": {"n_components": 33},

    "lgbm": {"learning_rate": 0.01080848474394827, "n_estimators": 3473, "num_leaves": 37,
             "min_child_samples": 5, "subsample": 0.6683039404912112,
             "colsample_bytree": 0.5594133344972811, "reg_lambda": 0.0016156274638098934,
             "reg_alpha": 1.4884784643707747},

    "xgb": {"learning_rate": 0.01597005288427683, "n_estimators": 3998, "max_depth": 5,
            "min_child_weight": 3.5840451018544606, "subsample": 0.8211950135665663,
            "colsample_bytree": 0.43687863673038857, "reg_lambda": 0.10444265249786544},

    "cat": {"learning_rate": 0.027624635538282816, "iterations": 2669, "depth": 7,
            "l2_leaf_reg": 1.7622393852341562},

    "gbr": {"n_estimators": 797, "learning_rate": 0.04331970465096853, "max_depth": 5,
            "min_samples_leaf": 17, "subsample": 0.6422862431579599,
            "max_features": 0.42865567424752304},

    "realmlp": {"lr": 0.020180527757062475, "wd": 0.00021803465714016984,
                "p_drop": 0.26401311250081483, "hidden_sizes": [512, 512, 512]},

    "krr": {"gamma": 0.006462318998350571, "alpha": 0.03998569559801853, "n_components": 2000},
}


def _p(family: str, defaults: dict, smoke_caps: dict = None) -> dict:

    out = dict(defaults)
    out.update(FAMILY_PARAMS.get(family, {}))
    if smoke_caps and AUTO_SMOKE:
        out.update(smoke_caps)
    return out


def feature_config() -> dict:

    return {"GEO_K": GEO_K, "NEIGHBOUR_K": NEIGHBOUR_K, "SMOOTHING": SMOOTHING}


def store_schema(family: str) -> str:

    payload = {
        "version": STORE_SCHEMA_VERSION,
        "family": family,
        "features": list(BASE_FEATURES) + list(EXTERNAL_FEATURES),
        "config": feature_config(),
        "params": FAMILY_PARAMS.get(family, {}),
    }
    blob = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:12]

ENSEMBLE = [
    ("hgb_deep", dict(max_iter=600, learning_rate=0.04, max_leaf_nodes=63,
                      min_samples_leaf=15, l2_regularization=1.0)),
    ("hgb_wide", dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                      min_samples_leaf=20, l2_regularization=1.0)),
    ("hgb_slow", dict(max_iter=900, learning_rate=0.02, max_leaf_nodes=31,
                      min_samples_leaf=30, l2_regularization=3.0)),
]

KNN_FEATURES = ["price_to_income", "log_income", "log_zhvi", "cost_ratio", "lat", "lon"]

FAMILIES = ["hgb", "forest", "mlp", "linear", "svr", "knn", "pls",
            "lgbm", "xgb", "cat", "gbr", "huber", "tabpfn",
            "tabicl", "tabdpt", "realmlp", "tabm", "krr"]

TABPFN_DEVICE = "auto"

ICL_MAX_ROWS = 10000

TABPFN_PIP = "tabpfn>=2,<3"

TABICL_PIP = "tabicl>=2.2,<3"

TABDPT_PIP = "tabdpt>=1.3,<2"

PYTABKIT_PIP = "pytabkit>=1.7,<2"


def _pip_import(module: str, spec: str | None):

    import importlib
    try:
        return importlib.import_module(module)
    except ImportError:
        if not spec:
            raise
    import subprocess
    print(f"  installing {spec} (needs Internet ON) ...")
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", spec],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"pip install {spec} failed: {r.stderr.strip()[-400:]}")
    importlib.invalidate_caches()
    mod = importlib.import_module(module)

    from importlib import metadata
    dist = spec.split(">")[0].split("<")[0].split("=")[0].strip()
    try:
        version = metadata.version(dist)
    except metadata.PackageNotFoundError:
        version = getattr(mod, "__version__", "?")
    print(f"  {module} {version} ready")
    return mod


def _free_gpu() -> None:

    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _require_cuda(family: str) -> None:

    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(f"{family} needs Accelerator = GPU (no CUDA device found)")


def _ensure_tabpfn():

    global TabPFNRegressor
    if TabPFNRegressor is None:
        TabPFNRegressor = _pip_import("tabpfn", TABPFN_PIP).TabPFNRegressor
    return TabPFNRegressor


def _per_zip(fit: pd.DataFrame, X: pd.DataFrame, y, seed: int, smoke: bool):

    zid = fit[ID].to_numpy()
    Xf = X.groupby(zid, sort=True).mean()
    yf = pd.Series(np.asarray(y, dtype=float), index=X.index).groupby(zid, sort=True).mean()
    if len(Xf) > ICL_MAX_ROWS:

        idx = np.random.default_rng(seed).choice(len(Xf), ICL_MAX_ROWS, replace=False)
        Xf, yf = Xf.iloc[idx], yf.iloc[idx]
    if smoke:
        Xf, yf = Xf.iloc[:200], yf.iloc[:200]
    return Xf, yf


def _design_hgb(fit: pd.DataFrame, targets: list[pd.DataFrame], features: list[str]):

    plain = [f for f in features if not _is_cat(f)]
    X = fit[plain].copy()
    Xs = [t[plain].copy() for t in targets]

    native = []
    for c in [f for f in features if _is_cat(f)]:
        if fit[c].nunique() <= NATIVE_CAT_MAX:
            levels = pd.Index(fit[c].astype(str).unique())
            X[c] = pd.Categorical(fit[c].astype(str), categories=levels)
            for t, Xt in zip(targets, Xs):
                Xt[c] = pd.Categorical(t[c].astype(str), categories=levels)
            native.append(c)
        else:
            encoded = _target_encode(fit, [fit] + targets, c)
            X[f"{c}_te"] = encoded[0].to_numpy()
            for Xt, e in zip(Xs, encoded[1:]):
                Xt[f"{c}_te"] = e.to_numpy()

    return X, Xs, [c in native for c in X.columns]

_DESIGN_CACHE = {}


def _design_numeric(fit: pd.DataFrame, targets: list[pd.DataFrame], features: list[str],
                    standardise: bool):

    key = (id(fit), len(fit), tuple(features), bool(standardise),
           tuple((id(t), len(t)) for t in targets))
    hit = _DESIGN_CACHE.get(key)
    if hit is not None:
        return hit

    plain = [f for f in features if not _is_cat(f)]
    X = fit[plain].copy()
    Xs = [t[plain].copy() for t in targets]

    for c in [f for f in features if _is_cat(f)]:
        encoded = _target_encode(fit, [fit] + targets, c)
        X[f"{c}_te"] = encoded[0].to_numpy()
        for Xt, e in zip(Xs, encoded[1:]):
            Xt[f"{c}_te"] = e.to_numpy()

    X = X.astype(float)
    Xs = [Xt[X.columns].astype(float) for Xt in Xs]

    X = X.replace([np.inf, -np.inf], np.nan)
    Xs = [Xt.replace([np.inf, -np.inf], np.nan) for Xt in Xs]

    med = X.median().fillna(0.0)
    X = X.fillna(med)
    Xs = [Xt.fillna(med) for Xt in Xs]

    if standardise:
        mu = X.mean()
        sd = X.std(ddof=0).replace(0.0, 1.0)
        X = (X - mu) / sd
        Xs = [(Xt - mu) / sd for Xt in Xs]

    X = X.astype(np.float32)
    Xs = [Xt.astype(np.float32) for Xt in Xs]

    _DESIGN_CACHE[key] = (X, Xs)
    return X, Xs

KEEP_FITTED = False

_FITTED = []

_USED_FEATURES = {}


def collect_fitted() -> list:

    out = list(_FITTED)
    _FITTED.clear()
    return out


def _keep(tag: str, model):

    if KEEP_FITTED:
        _FITTED.append((tag, model))
    return model

PSEUDO_FRAC = 0.10


def make_pseudo_rows(unlabelled: pd.DataFrame, preds: dict,
                     frac: float = None) -> pd.DataFrame:

    frac = PSEUDO_FRAC if frac is None else frac
    if frac <= 0 or not len(unlabelled) or len(preds) < 2:
        return unlabelled.iloc[:0].copy()

    P = np.column_stack([preds[f] for f in sorted(preds)])
    ok = np.isfinite(P).all(axis=1)
    if not ok.any():
        return unlabelled.iloc[:0].copy()

    disagree = np.where(ok, P.std(axis=1), np.inf)
    consensus = np.where(ok, P.mean(axis=1), np.nan)
    k = int(round(frac * ok.sum()))
    if k < 1:
        return unlabelled.iloc[:0].copy()

    keep = np.argsort(disagree)[:k]
    out = unlabelled.iloc[keep].copy()
    out[TARGET] = np.clip(consensus[keep], 0.0, 100.0)
    out["_pseudo"] = True
    return out


def _fit(model, X, y, w=None):

    if w is None:
        model.fit(X, y)
        return model
    try:
        model.fit(X, y, sample_weight=w)
    except TypeError:
        model.fit(X, y)
    return model


def _weights(fit: pd.DataFrame) -> np.ndarray | None:

    if "_pseudo" not in fit.columns or not bool((fit["_pseudo"] == True).any()):
        return None

    is_pseudo = (fit["_pseudo"] == True).to_numpy(dtype=bool)
    w = np.where(is_pseudo, PSEUDO_WEIGHT, 1.0)
    return w / float(np.mean(w))

PSEUDO_WEIGHT = 0.5


def _predict_family(family: str, fit: pd.DataFrame, targets: list[pd.DataFrame],
                    features: list[str], seed: int) -> list[np.ndarray]:

    assert "_pseudo" not in features, "_pseudo is a row marker, never a feature"

    seed = int(seed)
    y = fit[TARGET]
    w = _weights(fit)
    smoke = AUTO_SMOKE

    if family == "hgb":
        X, Xs, cat_mask = _design_hgb(fit, targets, features)
        members = ENSEMBLE[:1] if smoke else ENSEMBLE

        tuned = {k: v for k, v in FAMILY_PARAMS.get("hgb", {}).items()
                 if k not in ("max_iter", "learning_rate")}
        members = [(n, {**p, **tuned}) for n, p in members]
        if smoke:
            members = [(n, {**p, "max_iter": SMOKE_MAX_ITER}) for n, p in members]
        acc = [np.zeros(len(Xt)) for Xt in Xs]
        for _, params in members:
            model = _keep(f"hgb_{params['max_iter']}_{params['max_leaf_nodes']}",
                          HistGradientBoostingRegressor(
                              random_state=seed, early_stopping=False,
                              categorical_features=cat_mask, **params))
            _fit(model, X, y, w)
            for i, Xt in enumerate(Xs):
                acc[i] += model.predict(Xt[X.columns])
        return [a / len(members) for a in acc]

    if family == "forest":

        X, Xs = _design_numeric(fit, targets, features, standardise=False)
        n_trees = 20 if smoke else 400
        acc = [np.zeros(len(Xt)) for Xt in Xs]

        fp = _p("forest", dict(min_samples_leaf=5, rf_max_features=0.35, et_max_features=0.55))

        leaf = max(5, int(fp["min_samples_leaf"]))
        pair = [
            RandomForestRegressor(n_estimators=n_trees, min_samples_leaf=leaf,
                                  max_features=fp["rf_max_features"], random_state=seed,
                                  n_jobs=-1),

            ExtraTreesRegressor(n_estimators=n_trees, min_samples_leaf=leaf,
                                max_features=fp["et_max_features"], random_state=seed,
                                n_jobs=-1),
        ]
        for i, model in enumerate(pair):
            _keep(f"forest_{i}", model)
            _fit(model, X, y, w)
            for i, Xt in enumerate(Xs):
                acc[i] += model.predict(Xt[X.columns])
        return [a / len(pair) for a in acc]

    if family == "linear":

        X, Xs = _design_numeric(fit, targets, features, standardise=True)
        model = _keep("ridge", Ridge(**_p("linear", dict(alpha=10.0))))
        _fit(model, X, y, w)
        return [model.predict(Xt[X.columns]) for Xt in Xs]

    if family == "knn":

        feats = [f for f in KNN_FEATURES if f in fit.columns]
        if len(feats) < 3:
            return [np.full(len(t), float(y.mean())) for t in targets]
        X, Xs = _design_numeric(fit, targets, feats, standardise=True)
        kp = _p("knn", dict(n_neighbors=25, weights="distance", p=2))

        kp["n_neighbors"] = int(min(kp["n_neighbors"], len(X)))
        model = _keep("knn", KNeighborsRegressor(n_jobs=-1, **kp))
        _fit(model, X, y, w)
        return [model.predict(Xt[X.columns]) for Xt in Xs]

    if family == "mlp":

        X, Xs = _design_numeric(fit, targets, features, standardise=True)
        if smoke:
            nets = [MLPRegressor(hidden_layer_sizes=(32,), max_iter=12, random_state=seed)]
        else:
            mp = _p("mlp", dict(hidden_a=(128, 64), hidden_b=(64, 32), alpha_a=1e-3,
                                alpha_b=1e-2, lr=3e-3, batch_size=200))
            common = dict(max_iter=400, early_stopping=True, n_iter_no_change=15,
                          validation_fraction=0.12, learning_rate_init=mp["lr"],
                          batch_size=int(mp["batch_size"]))
            nets = [
                MLPRegressor(hidden_layer_sizes=tuple(mp["hidden_a"]), alpha=mp["alpha_a"],
                             random_state=seed, **common),
                MLPRegressor(hidden_layer_sizes=tuple(mp["hidden_b"]), alpha=mp["alpha_b"],
                             random_state=seed + 101, **common),
            ]
        acc = [np.zeros(len(Xt)) for Xt in Xs]
        for i, net in enumerate(nets):
            _keep(f"mlp_{i}", net)
            _fit(net, X, y, w)
            for i, Xt in enumerate(Xs):
                acc[i] += net.predict(Xt[X.columns])
        return [a / len(nets) for a in acc]

    if family == "svr":

        X, Xs = _design_numeric(fit, targets, features, standardise=True)
        model = _keep("linearsvr", LinearSVR(max_iter=5000, dual="auto", random_state=seed,
                                            **_p("svr", dict(C=1.0, epsilon=1.0))))
        _fit(model, X, y, w)
        return [model.predict(Xt[X.columns]) for Xt in Xs]

    if family == "pls":

        X, Xs = _design_numeric(fit, targets, features, standardise=True)
        n_comp = max(1, min(int(_p("pls", dict(n_components=12))["n_components"]),
                            X.shape[1] - 1))
        model = _keep("pls", PLSRegression(n_components=n_comp, scale=False))
        _fit(model, X, y, w)

        return [np.asarray(model.predict(Xt[X.columns])).ravel() for Xt in Xs]

    if family in ("lgbm", "xgb", "cat"):
        return _predict_boost(family, fit, targets, features, seed, smoke, w)

    if family == "gbr":

        X, Xs = _design_numeric(fit, targets, features, standardise=False)
        model = _keep("gbr", GradientBoostingRegressor(random_state=seed, **_p(
            "gbr", dict(n_estimators=500, learning_rate=0.05, max_depth=5, min_samples_leaf=15,
                        subsample=0.85), smoke_caps=dict(n_estimators=40))))
        _fit(model, X, y, w)
        return [model.predict(Xt[X.columns]) for Xt in Xs]

    if family == "tabpfn":

        cls = _ensure_tabpfn()
        if TABPFN_DEVICE == "auto":
            _require_cuda("tabpfn")
        X, Xs = _design_numeric(fit, targets, features, standardise=True)
        Xf, yf = _per_zip(fit, X, y, seed, smoke)

        model = _keep("tabpfn", cls(device=TABPFN_DEVICE, ignore_pretraining_limits=True,
                                    **_p("tabpfn", dict(n_estimators=8, softmax_temperature=0.9))))
        _fit(model, Xf, yf, None)
        return [np.asarray(model.predict(Xt[X.columns])).ravel() for Xt in Xs]

    if family == "huber":

        X, Xs = _design_numeric(fit, targets, features, standardise=True)
        model = _keep("huber", HuberRegressor(max_iter=500,
                                              **_p("huber", dict(epsilon=1.35, alpha=1e-3))))
        _fit(model, X, y, w)
        return [model.predict(Xt[X.columns]) for Xt in Xs]

    if family == "tabicl":

        mod = _pip_import("tabicl", TABICL_PIP)
        _require_cuda("tabicl")
        X, Xs = _design_numeric(fit, targets, features, standardise=True)
        Xf, yf = _per_zip(fit, X, y, seed, smoke)
        model = _keep("tabicl", mod.TabICLRegressor(random_state=seed, verbose=False, **_p(
            "tabicl", dict(n_estimators=8, outlier_threshold=4.0),
            smoke_caps=dict(n_estimators=1))))
        _fit(model, Xf, yf, None)
        return [np.asarray(model.predict(Xt[X.columns])).ravel() for Xt in Xs]

    if family == "tabdpt":

        mod = _pip_import("tabdpt", TABDPT_PIP)
        _require_cuda("tabdpt")
        X, Xs = _design_numeric(fit, targets, features, standardise=True)
        Xf, yf = _per_zip(fit, X, y, seed, smoke)
        dp = _p("tabdpt", dict(n_ensembles=8), smoke_caps=dict(n_ensembles=1))
        model = _keep("tabdpt", mod.TabDPTRegressor(context_reduction="subsample", compile=False,
                                                    use_flash=False, verbose=False))
        _fit(model, Xf.to_numpy(dtype=np.float32), yf.to_numpy(dtype=np.float32), None)

        return [np.asarray(model.predict(Xt[X.columns].to_numpy(dtype=np.float32),
                                         n_ensembles=int(dp["n_ensembles"]), context_size=None,
                                         batch_size=1024, seed=seed)).ravel() for Xt in Xs]

    if family in ("realmlp", "tabm"):

        mod = _pip_import("pytabkit", PYTABKIT_PIP)
        X, Xs = _design_numeric(fit, targets, features, standardise=True)
        cls = mod.RealMLP_TD_Regressor if family == "realmlp" else mod.TabM_D_Regressor
        kw = _p(family, {}, smoke_caps=dict(n_epochs=2))
        if "hidden_sizes" in kw:
            kw["hidden_sizes"] = [int(h) for h in kw["hidden_sizes"]]
        model = _keep(family, cls(random_state=seed, verbosity=0, **kw))
        _fit(model, X, y.to_numpy(dtype=np.float32), w)
        return [np.asarray(model.predict(Xt[X.columns])).ravel() for Xt in Xs]

    if family == "krr":

        X, Xs = _design_numeric(fit, targets, features, standardise=True)
        kp = _p("krr", dict(gamma=0.015, alpha=1.0, n_components=1000),
                smoke_caps=dict(n_components=50))
        ny = _keep("krr_nystroem", Nystroem(kernel="rbf", gamma=kp["gamma"], random_state=seed,
                                            n_components=min(int(kp["n_components"]), len(X))))
        Z = ny.fit_transform(X)
        model = _keep("krr_ridge", Ridge(alpha=kp["alpha"]))
        _fit(model, Z, y, w)
        return [model.predict(ny.transform(Xt[X.columns])) for Xt in Xs]

    raise ValueError(f"unknown family {family!r}")


def _predict_boost(family: str, fit: pd.DataFrame, targets: list[pd.DataFrame],
                   features: list[str], seed: int, smoke: bool,
                   w: np.ndarray = None) -> list[np.ndarray]:

    n_est = 40 if smoke else 600
    if family == "cat" and CatBoostRegressor is not None:
        cat_cols = [f for f in features if _is_cat(f)]
        plain = [f for f in features if not _is_cat(f)]
        X = fit[plain + cat_cols].copy()
        Xs = [t[plain + cat_cols].copy() for t in targets]
        for c in cat_cols:
            X[c] = X[c].astype(str)
            for Xt in Xs:
                Xt[c] = Xt[c].astype(str)
        X[plain] = X[plain].astype(float).fillna(X[plain].astype(float).median())
        for Xt in Xs:
            Xt[plain] = Xt[plain].astype(float).fillna(X[plain].median())
        model = _keep("catboost", CatBoostRegressor(
            verbose=0, random_seed=seed, allow_writing_files=False,
            cat_features=[X.columns.get_loc(c) for c in cat_cols],
            **_p("cat", dict(iterations=600, learning_rate=0.05, depth=6),
                 smoke_caps=dict(iterations=n_est))))
        _fit(model, X, fit[TARGET], w)
        return [np.asarray(model.predict(Xt[X.columns])).ravel() for Xt in Xs]

    if family == "lgbm" and lgb is not None:
        X, Xs = _design_numeric(fit, targets, features, standardise=False)
        model = _keep("lgbm", lgb.LGBMRegressor(
            random_state=seed, verbose=-1, **_p(
                "lgbm", dict(n_estimators=600, learning_rate=0.05, num_leaves=31,
                             min_child_samples=20, subsample=0.85, subsample_freq=1,
                             colsample_bytree=0.8), smoke_caps=dict(n_estimators=n_est))))
        _fit(model, X, fit[TARGET], w)
        return [model.predict(Xt[X.columns]) for Xt in Xs]

    if family == "xgb" and xgb is not None:
        X, Xs = _design_numeric(fit, targets, features, standardise=False)
        model = _keep("xgb", xgb.XGBRegressor(
            random_state=seed, verbosity=0, n_jobs=-1, **_p(
                "xgb", dict(n_estimators=600, learning_rate=0.05, max_depth=6,
                            min_child_weight=5, subsample=0.85, colsample_bytree=0.8,
                            reg_lambda=1.0), smoke_caps=dict(n_estimators=n_est))))
        _fit(model, X, fit[TARGET], w)
        return [model.predict(Xt[X.columns]) for Xt in Xs]

    raise RuntimeError(f"{family}: its library is not installed")

_FEATURE_CACHE = {}


def clear_feature_cache() -> None:
    _FEATURE_CACHE.clear()
    _DESIGN_CACHE.clear()


def build_features(fit: pd.DataFrame, targets: list[pd.DataFrame]):

    key = (id(fit), len(fit), tuple((id(t), len(t)) for t in targets))
    hit = _FEATURE_CACHE.get(key)
    if hit is not None:
        return hit

    f, *ts = impute_rent(fit, [fit] + list(targets))
    ext_cols = [c for c in EXTERNAL_FEATURES if c in f.columns]
    f, *ts = impute_external(f, [f] + ts, ext_cols)
    f = add_interactions(add_ratio_features(f))
    ts = [add_interactions(add_ratio_features(t)) for t in ts]
    f, *ts = add_neighbour_features(f, [f] + ts)

    f, *ts = add_geo_neighbour_features(f, [f] + ts)
    f, *ts = add_relative_features(f, [f] + ts)

    f, *ts = add_feature_neighbour_features(f, [f] + ts)

    _FEATURE_CACHE[key] = (f, ts)
    return f, ts


def fit_predict(fit: pd.DataFrame, targets: list[pd.DataFrame], seed: int,
                features: list[str] = None, family: str = "hgb") -> list[np.ndarray]:

    fit, targets = build_features(fit, list(targets))
    features = active_features(fit) if features is None else \
        [c for c in features if c in fit.columns]
    _USED_FEATURES[family] = list(features)
    return [np.clip(p, 0.0, 100.0) for p in _predict_family(family, fit, targets, features, seed)]


def persistence(history: pd.DataFrame, target_rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:

    latest = (history.sort_values(MONTH, key=lambda s: pd.to_datetime(s, format="%m/%d/%Y"))
                     .groupby(ID)[TARGET].last())
    by_state, glob = _measure_state_drift(history)
    pred = target_rows[ID].map(latest).to_numpy(dtype=float)
    if by_state and "State" in target_rows.columns:
        drift = target_rows["State"].map(by_state).fillna(glob).to_numpy(dtype=float)
    else:
        drift = glob
    return np.clip(pred + drift, 0.0, 100.0), ~np.isnan(pred)

DRIFT_SMOOTHING = 50.0


def _measure_drift(history: pd.DataFrame) -> float:

    if history[MONTH].nunique() < 2:
        return 0.0
    wide = history.pivot_table(index=ID, columns=MONTH, values=TARGET)
    if not set(TRAIN_MONTHS).issubset(wide.columns):
        return 0.0
    delta = (wide[TRAIN_MONTHS[1]] - wide[TRAIN_MONTHS[0]]).dropna()
    return float(delta.mean()) if len(delta) else 0.0


def _measure_state_drift(history: pd.DataFrame) -> tuple:

    glob = _measure_drift(history)
    if history[MONTH].nunique() < 2 or "State" not in history.columns:
        return {}, glob

    wide = history.pivot_table(index=ID, columns=MONTH, values=TARGET)
    if not set(TRAIN_MONTHS).issubset(wide.columns):
        return {}, glob
    delta = (wide[TRAIN_MONTHS[1]] - wide[TRAIN_MONTHS[0]]).dropna()

    state = history.drop_duplicates(ID).set_index(ID)["State"]
    df = pd.DataFrame({"d": delta}).join(state, how="inner")
    g = df.groupby("State")["d"].agg(["mean", "count"])
    shrunk = (g["mean"] * g["count"] + glob * DRIFT_SMOOTHING) / (g["count"] + DRIFT_SMOOTHING)
    return shrunk.to_dict(), glob


def _space(family: str, pick) -> dict:

    def arch(name, choices):
        return [int(x) for x in pick.c(name, choices).split("-")]

    if family == "lgbm":
        return dict(learning_rate=pick.f("learning_rate", 0.01, 0.1, log=True),
                    n_estimators=pick.i("n_estimators", 300, 4000),
                    num_leaves=pick.i("num_leaves", 15, 127),
                    min_child_samples=pick.i("min_child_samples", 5, 100),
                    subsample=pick.f("subsample", 0.5, 1.0),
                    colsample_bytree=pick.f("colsample_bytree", 0.4, 1.0),
                    reg_lambda=pick.f("reg_lambda", 1e-3, 10.0, log=True),
                    reg_alpha=pick.f("reg_alpha", 1e-3, 10.0, log=True))
    if family == "xgb":

        return dict(learning_rate=pick.f("learning_rate", 0.01, 0.1, log=True),
                    n_estimators=pick.i("n_estimators", 300, 4000),
                    max_depth=pick.i("max_depth", 3, 10),
                    min_child_weight=pick.f("min_child_weight", 1.0, 50.0, log=True),
                    subsample=pick.f("subsample", 0.5, 1.0),
                    colsample_bytree=pick.f("colsample_bytree", 0.4, 1.0),
                    reg_lambda=pick.f("reg_lambda", 1e-3, 10.0, log=True))
    if family == "cat":
        return dict(learning_rate=pick.f("learning_rate", 0.02, 0.15, log=True),
                    iterations=pick.i("iterations", 300, 3000),
                    depth=pick.i("depth", 4, 9),
                    l2_leaf_reg=pick.f("l2_leaf_reg", 1.0, 30.0, log=True))
    if family == "gbr":
        return dict(n_estimators=pick.i("n_estimators", 200, 1000),
                    learning_rate=pick.f("learning_rate", 0.02, 0.1, log=True),
                    max_depth=pick.i("max_depth", 3, 7),
                    min_samples_leaf=pick.i("min_samples_leaf", 5, 50),
                    subsample=pick.f("subsample", 0.6, 1.0),
                    max_features=pick.f("max_features", 0.3, 1.0))
    if family == "hgb":

        return dict(max_leaf_nodes=pick.i("max_leaf_nodes", 15, 127),
                    min_samples_leaf=pick.i("min_samples_leaf", 5, 60),
                    l2_regularization=pick.f("l2_regularization", 1e-3, 10.0, log=True),
                    max_bins=pick.i("max_bins", 63, 255))
    if family == "mlp":
        return dict(hidden_a=arch("hidden_a", ["64-32", "128-64", "256-128", "256-128-64"]),
                    hidden_b=arch("hidden_b", ["32-16", "64-32", "128-64"]),
                    alpha_a=pick.f("alpha_a", 1e-5, 1e-1, log=True),
                    alpha_b=pick.f("alpha_b", 1e-5, 1e-1, log=True),
                    lr=pick.f("lr", 5e-4, 1e-2, log=True),
                    batch_size=int(pick.c("batch_size", ["64", "128", "200", "256"])))
    if family == "forest":
        return dict(rf_max_features=pick.f("rf_max_features", 0.2, 0.7),
                    et_max_features=pick.f("et_max_features", 0.3, 0.9),
                    min_samples_leaf=pick.i("min_samples_leaf", 5, 20))

    if family == "linear":
        return dict(alpha=pick.f("alpha", 1e-2, 1e4, log=True))
    if family == "knn":
        return dict(n_neighbors=pick.i("n_neighbors", 5, 100),
                    weights=pick.c("weights", ["uniform", "distance"]),
                    p=int(pick.c("p", ["1", "2"])))
    if family == "svr":
        return dict(C=pick.f("C", 1e-3, 1e2, log=True),
                    epsilon=pick.f("epsilon", 0.0, 5.0))
    if family == "pls":
        return dict(n_components=pick.i("n_components", 2, 40))
    if family == "huber":
        return dict(epsilon=pick.f("epsilon", 1.1, 3.0),
                    alpha=pick.f("alpha", 1e-6, 10.0, log=True))
    if family == "tabpfn":

        return dict(n_estimators=int(pick.c("n_estimators", ["4", "8", "16"])),
                    softmax_temperature=pick.f("softmax_temperature", 0.5, 1.2))

    if family == "tabicl":
        return dict(n_estimators=int(pick.c("n_estimators", ["4", "8", "16"])),
                    outlier_threshold=pick.f("outlier_threshold", 2.5, 8.0))
    if family == "tabdpt":
        return dict(n_ensembles=int(pick.c("n_ensembles", ["4", "8", "16"])))
    if family == "realmlp":
        return dict(lr=pick.f("lr", 0.02, 0.4, log=True),
                    wd=pick.f("wd", 1e-4, 1e-1, log=True),
                    p_drop=pick.f("p_drop", 0.0, 0.4),
                    hidden_sizes=arch("hidden_sizes", ["256-256-256", "512-512-512", "256-256",
                                                       "512-256-128"]))
    if family == "tabm":
        return dict(lr=pick.f("lr", 3e-4, 5e-3, log=True),
                    weight_decay=pick.f("weight_decay", 1e-6, 1e-2, log=True),
                    dropout=pick.f("dropout", 0.0, 0.4),
                    d_block=int(pick.c("d_block", ["256", "512", "1024"])),
                    n_blocks=pick.i("n_blocks", 1, 4))
    if family == "krr":
        return dict(gamma=pick.f("gamma", 1e-3, 1e-1, log=True),
                    alpha=pick.f("alpha", 1e-3, 1e2, log=True),
                    n_components=int(pick.c("n_components", ["500", "1000", "2000"])))
    raise ValueError(f"no search space for {family!r}")


class _RandomPick:

    def __init__(self, rng):
        self.rng = rng

    def f(self, name, lo, hi, log=False):
        return float(np.exp(self.rng.uniform(np.log(lo), np.log(hi)))) if log \
            else float(self.rng.uniform(lo, hi))

    def i(self, name, lo, hi):
        return int(self.rng.integers(lo, hi + 1))

    def c(self, name, choices):
        return choices[int(self.rng.integers(len(choices)))]


class _OptunaPick:
    def __init__(self, trial):
        self.t = trial

    def f(self, name, lo, hi, log=False):
        return self.t.suggest_float(name, lo, hi, log=log)

    def i(self, name, lo, hi):
        return self.t.suggest_int(name, lo, hi)

    def c(self, name, choices):
        return self.t.suggest_categorical(name, choices)


def _tune_folds(train: pd.DataFrame, seed: int) -> list:

    folds = oof_folds(train, OOF_FOLDS, seed)
    jun = train[train[MONTH] == TRAIN_MONTHS[1]].reset_index(drop=True)
    jun_fold = oof_folds(jun, OOF_FOLDS, seed)
    return [(train[folds != f].reset_index(drop=True),
             jun[jun_fold == f].reset_index(drop=True))
            for f in range(OOF_FOLDS)]


def _cv_rmse(family: str, pairs: list, params: dict) -> float:

    saved = FAMILY_PARAMS.get(family)
    FAMILY_PARAMS[family] = params
    try:
        se, n = 0.0, 0
        for f, (fit, held) in enumerate(pairs):
            pred, = fit_predict(fit, [held], seed=f, family=family)
            err = held[TARGET].to_numpy(dtype=float) - pred
            se += float(np.sum(err * err))
            n += len(err)
            _FITTED.clear()
        return float(np.sqrt(se / n))
    finally:
        if saved is None:
            FAMILY_PARAMS.pop(family, None)
        else:
            FAMILY_PARAMS[family] = saved


def tune_family(fam: str):

    if fam not in FAMILIES:
        raise ValueError(f"RUN_FAMILY={fam!r} is not one of {FAMILIES}")
    t0 = time.time()
    budget = 60.0 * (1.0 if AUTO_SMOKE else TUNE_BUDGET_MIN)
    trials = 2 if AUTO_SMOKE else TUNE_MAX_TRIALS
    train, _, _ = load_all(verbose=True)
    current = dict(FAMILY_PARAMS.get(fam, {}))

    tune_pairs = _tune_folds(train, TUNE_SEED)
    base_t = _cv_rmse(fam, tune_pairs, current)
    per_trial = time.time() - t0
    print(f"\n=== {VERSION} W2 tuning: {fam} ===")
    print(f"  current params {current or '(hand-set defaults)'}")
    print(f"  baseline on tuning folds (seed {TUNE_SEED}): {base_t:.4f}  ({per_trial:.0f}s)")
    if trials == 0:

        print("  tuning OFF (TUNE_MAX_TRIALS = 0): storing with the current parameters")
        return False, current

    deadline = t0 + budget - 2.5 * per_trial
    history = []

    def objective(pick):
        params = _space(fam, pick)
        try:
            r = _cv_rmse(fam, tune_pairs, {**current, **params})
        except RuntimeError as exc:

            if "out of memory" not in str(exc).lower():
                raise
            _free_gpu()
            print(f"  trial {len(history) + 1:3d}  out of GPU memory, skipped: {params}")
            return float("inf")
        history.append((r, params))
        print(f"  trial {len(history):3d}  {r:.4f}  best {min(h[0] for h in history):.4f}")
        return r

    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="minimize",
                                    sampler=optuna.samplers.TPESampler(seed=TUNE_SEED))
        study.optimize(lambda t: objective(_OptunaPick(t)), n_trials=trials,
                       timeout=max(1.0, deadline - time.time()))
        engine = "optuna TPE"
    except ImportError:
        rng = np.random.default_rng(TUNE_SEED)
        while len(history) < trials and time.time() < deadline:
            objective(_RandomPick(rng))
        engine = "random search (optuna not installed)"

    if not history:
        print("  !! no trial finished inside the budget -- raise TUNE_BUDGET_MIN")
        return False, current
    best_t, best = min(history, key=lambda h: h[0])
    tuned = {**current, **best}

    stack_pairs = _tune_folds(train, OOF_SEED)
    base_s = _cv_rmse(fam, stack_pairs, current)
    best_s = _cv_rmse(fam, stack_pairs, tuned)
    gain_t, gain_s = base_t - best_t, base_s - best_s
    keep = gain_t >= TUNE_MIN_GAIN and gain_s >= TUNE_MIN_GAIN

    print(f"\n  {engine}, {len(history)} trials, {(time.time() - t0) / 60:.0f} min")
    print(f"  {'folds':28s} {'current':>9s} {'tuned':>9s} {'gain':>8s}")
    print(f"  {'tuning  (seed ' + str(TUNE_SEED) + ')':28s} {base_t:9.4f} {best_t:9.4f} {gain_t:+8.4f}")
    print(f"  {'store   (seed ' + str(OOF_SEED) + ')':28s} {base_s:9.4f} {best_s:9.4f} {gain_s:+8.4f}")
    print(f"\n  verdict: {'KEEP' if keep else 'REJECT'} (rule: >= {TUNE_MIN_GAIN} on BOTH)")

    path = get_output_dir() / f"tuned_{fam}.json"
    path.write_text(json.dumps({"family": fam, "keep": keep, "params": tuned,
                                "tuning": [base_t, best_t], "store": [base_s, best_s],
                                "trials": len(history), "engine": engine}, indent=1))
    print(f"  wrote {path}")
    if keep:
        print("\n  paste into FAMILY_PARAMS in src/train.py:")
        print(f"    {fam!r}: {json.dumps(tuned)},")
    return keep, tuned


def check_submission(sub: pd.DataFrame, sample: pd.DataFrame) -> None:

    assert list(sub.columns) == list(sample.columns), f"columns {list(sub.columns)}"
    assert len(sub) == len(sample), f"{len(sub)} rows, expected {len(sample)}"
    assert (sub[ID].to_numpy() == sample[ID].to_numpy()).all(), "Zip order differs from sample"
    assert sub[ID].str.len().eq(5).all(), "a Zip lost its leading zero"
    assert sub[ID].duplicated().sum() == 0, "duplicate Zip"
    v = sub[TARGET].to_numpy(dtype=float)
    assert np.isfinite(v).all(), "non-finite prediction"
    assert (v >= 0).all() and (v <= 100).all(), f"out of range: [{v.min()}, {v.max()}]"
    print(f"  submission OK: {len(sub)} rows, range [{v.min():.2f}, {v.max():.2f}], "
          f"mean {v.mean():.2f}")


def run_one_family(fam: str = None) -> None:

    fam = fam or RUN_FAMILY
    if fam not in FAMILIES:
        raise ValueError(f"RUN_FAMILY={fam!r} is not one of {FAMILIES}")
    global KEEP_FITTED
    KEEP_FITTED = True
    print(f"\n=== {VERSION} family run: {fam} ===")
    t0 = time.time()
    train, test, _ = load_all(verbose=True)
    res = build_family_oof(fam, train, test, fit_predict,
                           clear_feature_cache, collect=collect_fitted)
    used = _USED_FEATURES.get(fam, [])
    pred_path = save_family(
        fam, res["oof"], res["test"], res["zips"], res["test_zips"],
        res["fingerprint"], meta={"rmse": res["rmse"], "version": VERSION,
                                  "n_models": len(res["models"]),
                                  "params": FAMILY_PARAMS.get(fam, {}),
                                  "n_features": len(used), "features": used,
                                  "minutes": round((time.time() - t0) / 60, 1),
                                  "env": _env()},
        schema=store_schema(fam))
    bundle_zip(fam, pred_path, res["models"])
    print(f"\n{fam}: OOF RMSE {res['rmse']:.4f}, {len(res['models'])} estimators")
    print(f"  features reaching the model: {len(used)}")
    print(f"  schema {store_schema(fam)}  params {json.dumps(FAMILY_PARAMS.get(fam, {}))}")
    print(f"  download {fam}_store.zip -- it goes into the V9 model_weights dataset")


def _env() -> dict:

    from importlib import metadata
    out = {"python": sys.version.split()[0]}
    for lib in ("numpy", "pandas", "scikit-learn", "lightgbm", "xgboost", "catboost",
                "optuna", "tabpfn", "torch", "tabicl", "tabdpt", "pytabkit", "faiss-cpu"):
        try:
            out[lib] = metadata.version(lib)
        except metadata.PackageNotFoundError:
            pass
    return out

LEVEL2 = "ridge"
LEVEL2_META = False


def _level2_models():

    return {
        "ridge": lambda: Ridge(alpha=1.0),
        "hgb": lambda: HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.05, max_depth=3,
            min_samples_leaf=40, l2_regularization=1.0,
            early_stopping=False, random_state=0),
    }


def blend_from_store() -> None:

    train, test, sample = load_all(verbose=True)
    fp = oof_fingerprint(train)
    print(f"\n=== {VERSION} blend from store (fingerprint {fp}) ===")
    stored = load_families(fp, schema_fn=store_schema)
    if not stored:
        print("  !! no stored families found")
        return
    left_out = sorted(set(stored) - set(SHIP_FAMILIES))
    missing = [f for f in SHIP_FAMILIES if f not in stored]
    stored = {f: stored[f] for f in SHIP_FAMILIES if f in stored}
    if left_out:
        print(f"  stored but not in SHIP_FAMILIES, left out: {', '.join(left_out)}")
    if missing:
        print(f"  !! in SHIP_FAMILIES but no valid store: {', '.join(missing)} -- "
              f"this is NOT the intended submission")

    jun = train[train[MONTH] == TRAIN_MONTHS[1]].reset_index(drop=True)
    y = jun[TARGET].to_numpy(dtype=float)
    folds = oof_folds(jun).to_numpy()
    oof = {f: align(p, jun[ID], "oof") for f, p in stored.items()}
    tst = {f: align(p, test[ID], "test") for f, p in stored.items()}
    fams = sorted(oof)

    print(f"\n  {len(fams)} families, OOF RMSE:")
    for f in sorted(fams, key=lambda k: np.sqrt(np.nanmean((y - oof[k]) ** 2))):
        print(f"    {f:10s} {np.sqrt(np.nanmean((y - oof[f]) ** 2)):.4f}")

    fitted = optimise_weights(oof, y, fams)
    print("\n  Level-1 fitted weights (diagnostic): "
          + ", ".join(f"{f} {fitted[f]:.3f}" for f in fams if fitted[f] > 0.0005))
    print(f"    fitted {blend_rmse(oof, y, fitted):.4f}   "
          f"equal {blend_rmse(oof, y, {f: 1.0 / len(fams) for f in fams}):.4f}")

    print("\n  === LEVEL 2: nested on the same folds ===")
    l2 = {}
    l2_oof = {}
    for name, maker in _level2_models().items():
        for extra in (False, True):
            tag = f"{name}{'+meta' if extra else ''}"
            pred = level2_oof(oof, y, folds, fams, maker, extra=extra)
            ok = np.isfinite(pred)
            l2[tag] = float(np.sqrt(np.mean((y[ok] - np.clip(pred[ok], 0, 100)) ** 2)))
            l2_oof[tag] = np.clip(pred, 0.0, 100.0)
            print(f"    {tag:22s} {l2[tag]:.4f}")

    maker = _level2_models()[LEVEL2]
    tag = f"{LEVEL2}{'+meta' if LEVEL2_META else ''}"
    best = min(l2, key=l2.get)
    print(f"\n  submitting LEVEL 2: {tag}  (scored {l2[tag]:.4f})")
    if tag != best:
        print(f"  NOTE: {best} scored better ({l2[best]:.4f}); shipping {tag} by explicit setting")
    model_pred = np.clip(level2_fit_predict(oof, y, tst, fams, maker, extra=LEVEL2_META), 0.0, 100.0)

    new_tst = _pseudo_test_preds(train, test, model_pred, tst)
    if new_tst is not None:
        p2 = np.clip(level2_fit_predict(oof, y, new_tst, fams, maker, extra=LEVEL2_META),
                     0.0, 100.0)
        shift = float(np.sqrt(np.mean((p2 - model_pred) ** 2)))
        print(f"  re-blended through {LEVEL2}: shipped prediction moved {shift:.4f} RMS")
        model_pred = p2

    if NBR_RESID_K:
        model_pred = _nbr_resid_correct(jun, y - l2_oof[tag], test, model_pred, NBR_RESID_K)

    carried, has_history = persistence(train, test)
    pred = np.where(has_history, carried, model_pred)
    print(f"  test rows {len(test)}: {int(has_history.sum())} seen (persistence), "
          f"{int((~has_history).sum())} new (blend)")

    sub = sample.copy()
    sub[TARGET] = np.clip(pred, 0.0, 100.0)
    check_submission(sub, sample)
    path = get_output_dir() / "submission.csv"
    sub.to_csv(path, index=False)
    print(f"  wrote {path}")


def _nbr_resid_correct(fit_rows: pd.DataFrame, resid: np.ndarray, target_rows: pd.DataFrame,
                       pred: np.ndarray, k: int) -> np.ndarray:

    from sklearn.neighbors import BallTree
    have = np.isfinite(fit_rows["lat"].to_numpy(float)) & np.isfinite(fit_rows["lon"].to_numpy(float))
    have &= np.isfinite(resid)
    pts = np.radians(fit_rows.loc[have, ["lat", "lon"]].to_numpy(float))
    r = resid[have]
    tree = BallTree(pts, metric="haversine")
    _, idx = tree.query(pts, k=k + 1)
    nb = r[idx[:, 1:]].mean(axis=1)
    slope = float(np.clip(np.cov(nb, r)[0, 1] / max(np.var(nb, ddof=1), 1e-12), 0.0, 1.0))
    tq = np.isfinite(target_rows["lat"].to_numpy(float)) & np.isfinite(target_rows["lon"].to_numpy(float))
    _, it = tree.query(np.radians(target_rows.loc[tq, ["lat", "lon"]].to_numpy(float)), k=k)
    shift = slope * r[it].mean(axis=1)
    out = pred.copy()
    out[tq] = np.clip(out[tq] + shift, 0.0, 100.0)
    print(f"  neighbour residual correction: k={k}, slope {slope:.3f} "
          f"(corr {np.corrcoef(nb, r)[0, 1]:+.3f}), shift RMS {np.sqrt(np.mean(shift ** 2)):.3f}")
    return out


def _pseudo_test_preds(train: pd.DataFrame, test: pd.DataFrame,
                       store_pred: np.ndarray, tst: dict) -> dict:

    usable = {k: v for k, v in tst.items() if np.isfinite(v).any()}
    if len(usable) < 2:
        print("  pseudo-labels: fewer than two stored families, skipped")
        return None

    extra = make_pseudo_rows(test, usable, frac=PSEUDO_FRAC)
    if not len(extra):
        print("  pseudo-labels: none selected, shipping the V7 blend unchanged")
        return None

    pos = {z: i for i, z in enumerate(test[ID].to_numpy())}
    take = np.array([pos[z] for z in extra[ID].to_numpy()])
    extra[TARGET] = np.clip(store_pred[take], 0.0, 100.0)

    aug = pd.concat([train, extra], ignore_index=True)
    feats = None
    fams = sorted(tst)
    print(f"  pseudo-labels: +{len(extra)} rows from {len(usable)} stored families "
          f"(round 1 free), refitting {len(fams)} families on {len(aug)} rows")

    out = {}
    for fam in fams:
        clear_feature_cache()
        try:
            pr, = fit_predict(aug, [test], 0, features=feats, family=fam)
            out[fam] = np.clip(pr, 0.0, 100.0)
        except Exception as exc:

            print(f"    !! {fam} refit failed ({type(exc).__name__}: {exc}) -- keeping V7's")
            out[fam] = tst[fam]
        _FITTED.clear()
        gc.collect()

    clear_feature_cache()
    moved = {f: float(np.sqrt(np.nanmean((out[f] - tst[f]) ** 2))) for f in fams}
    worst = max(moved, key=moved.get)
    print(f"    per-family RMS shift: median {np.median(list(moved.values())):.4f}, "
          f"max {moved[worst]:.4f} ({worst})")
    return out

RUN_LOG = "run_log.txt"


class _Tee:

    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, s):
        if not self.log.closed:
            self.log.write(s)
        return self.stream.write(s)

    def flush(self):
        if not self.log.closed:
            self.log.flush()
        self.stream.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def _drop_outputs(fam: str) -> None:

    out = get_output_dir()
    for p in (out / f"{fam}_store.zip", out / f"tuned_{fam}.json", store_dir() / f"{fam}.json"):
        p.unlink(missing_ok=True)


def _pack_logs(fam: str) -> None:

    out = get_output_dir()
    zip_path = out / f"{fam}_store.zip"
    if not zip_path.exists():
        return
    with zipfile.ZipFile(zip_path, "a", zipfile.ZIP_DEFLATED) as zf:
        for src, name in ((out / RUN_LOG, f"{fam}_run_log.txt"),
                          (out / f"tuned_{fam}.json", f"tuned_{fam}.json")):
            if src.exists():
                zf.write(src, arcname=f"logs/{name}")
    print(f"  packed logs/ into {zip_path.name} ({zip_path.stat().st_size / 1e6:.1f} MB)")


def main() -> None:

    real = sys.stdout, sys.stderr

    with open(get_output_dir() / RUN_LOG, "w", buffering=1) as log:
        sys.stdout, sys.stderr = _Tee(real[0], log), _Tee(real[1], log)
        try:
            _run()
        except BaseException:
            import traceback
            traceback.print_exc(file=log)
            raise
        finally:
            sys.stdout, sys.stderr = real
    if RUN_FAMILY is not None:
        _pack_logs(RUN_FAMILY)


def _run() -> None:
    global AUTO_SMOKE

    if RUN_FAMILY is not None:
        fam = RUN_FAMILY

        def tune_then_store():

            global KEEP_FITTED
            KEEP_FITTED = False
            keep, tuned = tune_family(fam)
            if keep:
                FAMILY_PARAMS[fam] = tuned
            print(f"\n=== storing {fam} with {'the TUNED' if keep else 'its CURRENT'} parameters ===")
            clear_feature_cache()
            _FITTED.clear()
            run_one_family(fam)

        if AUTO_SMOKE:
            print("########## SMOKE ##########")
            saved = {k: dict(v) for k, v in FAMILY_PARAMS.items()}
            tune_then_store()

            FAMILY_PARAMS.clear()
            FAMILY_PARAMS.update(saved)
            AUTO_SMOKE = False
            clear_feature_cache()
            _FITTED.clear()
            gc.collect()
            _drop_outputs(fam)
            print("########## SMOKE OK -- real run follows ##########")
        tune_then_store()
        return

    if AUTO_SMOKE:
        print("########## SMOKE ##########")
        blend_from_store()
        AUTO_SMOKE = False
        clear_feature_cache()
        _FITTED.clear()
        gc.collect()
        print("########## SMOKE OK -- real run follows ##########")
    blend_from_store()

if __name__ == "__main__":
    main()
