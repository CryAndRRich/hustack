import time

import lightgbm as lgb
import numpy as np
import optuna
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer

from preprocess import BINARY, NUMERIC, ORDINAL

optuna.logging.set_verbosity(optuna.logging.WARNING)

CATS = list(ORDINAL) + list(BINARY)
GAM_ALPHAS = np.logspace(-3, 3, 13)
GAM_BASE = {"degree": 3, "knot_strategy": "quantile", "alpha": None, **{f"knots_{c}": 12 for c in NUMERIC}}
LGB_BASE = {
    "learning_rate": 0.02, "num_leaves": 15, "max_depth": 4, "min_child_samples": 40,
    "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0, "reg_alpha": 0.0,
    "min_split_gain": 0.0, "path_smooth": 0.0, "max_bin": 255, "linear_tree": False,
}
LGB_FIXED = {"objective": "regression", "subsample_freq": 1, "verbose": -1, "n_jobs": -1}


def build_gam(p):
    blocks = [
        (f"spl_{c}", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("spl", SplineTransformer(n_knots=p[f"knots_{c}"], degree=p["degree"], knots=p["knot_strategy"],
                                      extrapolation=p.get("extrapolation", "linear"))),
        ]), [c])
        for c in NUMERIC
    ]
    blocks.append(("miss", MissingIndicator(features="all"), ["sleep_hours", "screen_time_hours"]))
    blocks.append(("cat", Pipeline([
        ("imp", SimpleImputer(strategy="constant", fill_value=-1)),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]), CATS))
    ridge = RidgeCV(alphas=GAM_ALPHAS) if p["alpha"] is None else Ridge(alpha=p["alpha"])
    return Pipeline([("design", ColumnTransformer(blocks)), ("ridge", ridge)])


def fit_gam(p, X, y):
    return build_gam(p).fit(X, y)


def fit_lgb(p, X, y, cfg, seed):
    X_fit, X_es, y_fit, y_es = train_test_split(X, y, test_size=0.1, random_state=seed)
    model = lgb.LGBMRegressor(n_estimators=cfg["max_rounds"], random_state=seed, **LGB_FIXED, **p)
    model.fit(X_fit, y_fit, eval_set=[(X_es, y_es)],
              callbacks=[lgb.early_stopping(cfg["early_stop"], verbose=False)])
    return model


def suggest_gam(trial):
    p = {f"knots_{c}": trial.suggest_int(f"knots_{c}", 3, 20) for c in NUMERIC}
    p["degree"] = trial.suggest_int("degree", 1, 3)
    p["knot_strategy"] = trial.suggest_categorical("knot_strategy", ["quantile", "uniform"])
    p["alpha"] = trial.suggest_float("alpha", 1e-4, 1e3, log=True)
    return p


def suggest_lgb(trial):
    p = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 4, 64, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 200, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.5),
        "path_smooth": trial.suggest_float("path_smooth", 0.0, 20.0),
        "max_bin": trial.suggest_categorical("max_bin", [63, 127, 255, 511]),
        "linear_tree": trial.suggest_categorical("linear_tree", [False, True]),
    }
    if p["linear_tree"]:
        p["linear_lambda"] = trial.suggest_float("linear_lambda", 1e-3, 10.0, log=True)
    return p


def cv_rmse(family, p, X, y, cfg, seed):
    pred = np.zeros(len(y))
    for k, (tr, va) in enumerate(KFold(cfg["inner_folds"], shuffle=True, random_state=seed).split(X)):
        if family == "gam":
            model = fit_gam(p, X.iloc[tr], y[tr])
        else:
            model = fit_lgb(p, X.iloc[tr], y[tr], cfg, seed + k)
        pred[va] = model.predict(X.iloc[va])
    return float(np.sqrt(np.mean((y - pred) ** 2)))


def tune(family, X, y, cfg, seed):
    suggest = suggest_gam if family == "gam" else suggest_lgb
    base = dict(GAM_BASE, alpha=0.01) if family == "gam" else dict(LGB_BASE, reg_alpha=1e-3)

    def objective(trial):
        p = suggest(trial)
        try:
            return cv_rmse(family, p, X, y, cfg, seed)
        except ValueError as e:
            trial.set_user_attr("error", str(e)[:200])
            return float("inf")

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=seed, multivariate=True))
    study.enqueue_trial({k: v for k, v in base.items() if k != "linear_lambda"})
    t0 = time.time()
    study.optimize(objective, n_trials=cfg[f"{family}_trials"], timeout=cfg[f"{family}_timeout"])
    best = suggest(optuna.trial.FixedTrial(study.best_params))
    n_err = sum("error" in t.user_attrs for t in study.trials)
    print(f"    tune {family}: {len(study.trials)} trials in {time.time() - t0:.0f}s  "
          f"incumbent {study.trials[0].value:.4f} -> best {study.best_value:.4f}  errors {n_err}")
    return best, study
