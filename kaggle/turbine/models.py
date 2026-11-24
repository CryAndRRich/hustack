import time

import lightgbm as lgb
import numpy as np

import config
from runtime import log, step

MODELS = {
    "tweedie": ("tweedie", "tweedie", "rev", "sw", "all"),
    "tweedie_w": ("tweedie", "tweedie", "rev", "sw_iw", "all"),
    "payer": ("binary", "binary_logloss", "payer", "sw", "all"),
    "install": ("binary", "binary_logloss", "inst", "sw", "all"),
    "gamma": ("gamma", "gamma", "rev", "sw", "payers"),
}


def progress(period=200):
    def callback(env):
        if (env.iteration + 1) % period == 0 and env.evaluation_result_list:
            _data, metric, value = env.evaluation_result_list[0][:3]
            log("    iteration %d  %s %.6f" % (env.iteration + 1, metric, value))
    return callback


def params(objective, metric):
    p = dict(config.LGB_BASE, objective=objective, metric=metric)
    if objective == "tweedie":
        p["tweedie_variance_power"] = config.TWEEDIE_POWER
    return p


def columns(t, name):
    _obj, _met, label, weight, rows = MODELS[name]
    y = {"rev": t["rev"], "payer": (t["rev"] > 0).astype(np.float64), "inst": t["inst"].astype(np.float64)}[label]
    w = t["sw"] * t["iw"] if weight == "sw_iw" else t["sw"]
    idx = np.flatnonzero(t["rev"] > 0) if rows == "payers" else None
    return y, w, idx


def fit_holdout(name, X, t, n_tr, names, rounds):
    obj, met = MODELS[name][:2]
    y, w, idx = columns(t, name)
    if idx is None:
        Xa, ya, wa, Xb, yb, wb = X[:n_tr], y[:n_tr], w[:n_tr], X[n_tr:], y[n_tr:], w[n_tr:]
    else:
        a, b = idx[idx < n_tr], idx[idx >= n_tr]
        Xa, ya, wa, Xb, yb, wb = X[a], y[a], w[a], X[b], y[b], w[b]
    t0 = time.time()
    dtr = lgb.Dataset(Xa, ya, weight=wa, feature_name=names)
    dva = lgb.Dataset(Xb, yb, weight=wb, reference=dtr)
    model = lgb.train(params(obj, met), dtr, rounds, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(config.PATIENCE, verbose=False), progress()])
    best = max(model.best_iteration, 1)
    step("%s holdout fit: %d rows, best iteration %d, %.0fs" % (name, len(ya), best, time.time() - t0))
    return model, best


def fit_full(name, X, t, names, rounds):
    obj, met = MODELS[name][:2]
    y, w, idx = columns(t, name)
    if idx is not None:
        X, y, w = X[idx], y[idx], w[idx]
    t0 = time.time()
    model = lgb.train(params(obj, met), lgb.Dataset(X, y, weight=w, feature_name=names), rounds)
    step("%s refit on all rows: %d rows, %d rounds, %.0fs" % (name, len(y), rounds, time.time() - t0))
    return model


def arm_scores(preds):
    return {"tweedie": preds["tweedie"], "tweedie_w": preds["tweedie_w"], "payer": preds["payer"],
            "install": preds["install"], "hurdle": preds["payer"] * preds["gamma"]}


def importance(model, names):
    gain = model.feature_importance("gain")
    return sorted(zip(names, gain.tolist()), key=lambda t: -t[1])


def log_top(model, names, title, n=25):
    log(title)
    for nm, g in importance(model, names)[:n]:
        log("  %-50s %14.1f" % (nm[:50], g))
