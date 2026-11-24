import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_auc_score

import config
from models import progress
from runtime import log, step


def fit(Xtr, sw, Xca, names, rounds):
    rng = np.random.default_rng(config.SEED + 1)
    itr = rng.choice(len(Xtr), min(config.ADV_ROWS, len(Xtr)), replace=False)
    ica = rng.choice(len(Xca), min(config.ADV_ROWS, len(Xca)), replace=False)
    X = np.vstack([Xtr[itr], Xca[ica]])
    y = np.r_[np.zeros(len(itr)), np.ones(len(ica))]
    w = np.r_[sw[itr], np.full(len(ica), sw[itr].sum() / len(ica))]
    hold = rng.random(len(y)) < 0.2
    params = dict(config.LGB_BASE, objective="binary", metric="auc", learning_rate=0.1)
    dtr = lgb.Dataset(X[~hold], y[~hold], weight=w[~hold], feature_name=names)
    dva = lgb.Dataset(X[hold], y[hold], weight=w[hold], reference=dtr)
    model = lgb.train(params, dtr, rounds, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(30, verbose=False), progress(100)])
    auc = roc_auc_score(y[hold], model.predict(X[hold]), sample_weight=w[hold])
    del X, dtr, dva
    step("adversarial fitted: %d rounds, holdout AUC %.4f" % (model.best_iteration, auc))

    p = np.clip(model.predict(Xtr), 1e-6, 1 - 1e-6)
    iw = p / (1 - p)
    iw = np.minimum(iw, np.quantile(iw, config.IW_CLIP_Q))
    iw = iw / np.average(iw, weights=sw)
    ess = (sw * iw).sum() ** 2 / (sw * sw * iw * iw).sum() / ((sw.sum() ** 2) / (sw * sw).sum())
    gain = model.feature_importance("gain")
    top = sorted(zip(names, gain), key=lambda t: -t[1])[:25]
    log("adversarial AUC %.4f; importance weights: median %.3f p99 %.3f max %.3f; relative ESS %.3f"
        % (auc, np.median(iw), np.quantile(iw, 0.99), iw.max(), ess))
    log("features that separate training from candidates (gain):")
    for n, g in top:
        log("  %-50s %12.0f" % (n[:50], g))
    report = {"auc": auc, "rounds": model.best_iteration, "relative_ess": ess,
              "iw_median": float(np.median(iw)), "iw_max": float(iw.max()),
              "top_features": [[n, float(g)] for n, g in top]}
    return iw, report
