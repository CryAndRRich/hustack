import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import f1_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from preprocess import TARGET, features, load

VERSION = "V1"
OUT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
FOLDS = 5
REPEATS = 3
INNER_VAL = 0.15
BUDGET = {"lr": 900, "lgb": 1800, "xgb": 1800, "cat": 3600, "realmlp": 3600}
SMOKE_ROWS = 1500
SMOKE_ONLY = os.environ.get("SMOKE_ONLY") == "1"
N_JOBS = os.cpu_count() or 4


def inner_split(y, seed):
    return train_test_split(np.arange(len(y)), test_size=INNER_VAL, stratify=y, random_state=seed)


def cat_cols(X):
    return [c for c in X.columns if isinstance(X[c].dtype, pd.CategoricalDtype)]


def fit_lr(X, y, Xs, seed, smoke):
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, QuantileTransformer, SplineTransformer, StandardScaler
    cats = cat_cols(X)
    nums = [c for c in X.columns if c not in cats]
    num_pipe = make_pipeline(QuantileTransformer(n_quantiles=200, random_state=seed),
                             SplineTransformer(n_knots=5, degree=3), StandardScaler())
    ct = ColumnTransformer([("num", num_pipe, nums), ("cat", OneHotEncoder(handle_unknown="ignore"), cats)])
    m = make_pipeline(ct, LogisticRegression(C=0.1, max_iter=200 if smoke else 3000))
    m.fit(X, y)
    return [m.predict_proba(Z)[:, 1] for Z in Xs]


def fit_lgb(X, y, Xs, seed, smoke):
    import lightgbm as lgb
    a, b = inner_split(y, seed)
    m = lgb.LGBMClassifier(n_estimators=50 if smoke else 5000, learning_rate=0.02, num_leaves=15,
                           min_child_samples=30, subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
                           reg_lambda=1.0, cat_smooth=20, min_data_per_group=30, random_state=seed,
                           n_jobs=N_JOBS, verbose=-1)
    m.fit(X.iloc[a], y[a], eval_set=[(X.iloc[b], y[b])], callbacks=[lgb.early_stopping(200, verbose=False)])
    return [m.predict_proba(Z)[:, 1] for Z in Xs]


def fit_xgb(X, y, Xs, seed, smoke):
    import xgboost as xgb
    a, b = inner_split(y, seed)
    m = xgb.XGBClassifier(n_estimators=50 if smoke else 5000, learning_rate=0.02, max_depth=4,
                          min_child_weight=3, subsample=0.8, colsample_bytree=0.7, reg_lambda=1.0,
                          tree_method="hist", enable_categorical=True, max_cat_to_onehot=8,
                          early_stopping_rounds=200, eval_metric="logloss", random_state=seed, n_jobs=N_JOBS)
    m.fit(X.iloc[a], y[a], eval_set=[(X.iloc[b], y[b])], verbose=False)
    return [m.predict_proba(Z)[:, 1] for Z in Xs]


def fit_cat(X, y, Xs, seed, smoke):
    from catboost import CatBoostClassifier
    cats = cat_cols(X)

    def prep(Z):
        Z = Z.copy()
        Z[cats] = Z[cats].astype(str)
        return Z
    a, b = inner_split(y, seed)
    m = CatBoostClassifier(iterations=50 if smoke else 5000, learning_rate=0.03, depth=6, l2_leaf_reg=3,
                           random_seed=seed, eval_metric="Logloss", early_stopping_rounds=300,
                           use_best_model=True, thread_count=N_JOBS, verbose=0, cat_features=cats)
    m.fit(prep(X.iloc[a]), y[a], eval_set=(prep(X.iloc[b]), y[b]))
    return [m.predict_proba(prep(Z))[:, 1] for Z in Xs]


def fit_realmlp(X, y, Xs, seed, smoke):
    from pytabkit import RealMLP_TD_Classifier
    kw = {"n_epochs": 4} if smoke else {}
    m = RealMLP_TD_Classifier(random_state=seed, n_threads=N_JOBS, device="cpu", verbosity=0,
                              val_metric_name="cross_entropy", **kw)
    m.fit(X, y)
    return [m.predict_proba(Z)[:, 1] for Z in Xs]


ARMS = {"lr": fit_lr, "lgb": fit_lgb, "xgb": fit_xgb, "cat": fit_cat, "realmlp": fit_realmlp}


def install_pytabkit():
    if importlib.util.find_spec("pytabkit") is not None:
        return
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pytabkit"], capture_output=True, text=True)
    print("pip install pytabkit:", "ok" if r.returncode == 0 else f"failed ({r.stderr.strip()[-300:]})")


def on_alarm(signum, frame):
    raise TimeoutError("family time budget reached")


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def fit_stack(L, y):
    m = L.shape[1]

    def loss(w):
        z = L @ w[:m] + w[m]
        return np.mean(np.logaddexp(0, z) - y * z)

    def grad(w):
        s = sigmoid(L @ w[:m] + w[m]) - y
        return np.r_[L.T @ s, s.sum()] / len(y)
    w0 = np.r_[np.full(m, 1 / m), 0.0]
    r = minimize(loss, w0, jac=grad, method="L-BFGS-B", bounds=[(0, None)] * m + [(None, None)])
    return r.x


def apply_stack(w, L):
    return sigmoid(L @ w[:-1] + w[-1])


def best_threshold(y, p):
    o = np.argsort(-p, kind="stable")
    ps, tp = p[o], np.cumsum(y[o])
    f1 = 2 * tp / (np.arange(1, len(y) + 1) + y.sum())
    i = int(np.argmax(f1))
    thr = ps[i] - 1e-12 if i + 1 == len(ps) else 0.5 * (ps[i] + ps[i + 1])
    return float(f1[i]), float(thr)


def plugin_labels(p):
    o = np.argsort(-p, kind="stable")
    ef1 = 2 * np.cumsum(p[o]) / (np.arange(1, len(p) + 1) + p.sum())
    out = np.zeros(len(p), dtype=int)
    out[o[:int(np.argmax(ef1)) + 1]] = 1
    return out


def fold_f1(y, p, fold):
    thr, plug = [], []
    for k in range(FOLDS):
        v = fold == k
        _, t = best_threshold(y[~v], p[~v])
        thr.append(f1_score(y[v], (p[v] > t).astype(int)))
        plug.append(f1_score(y[v], plugin_labels(p[v])))
    return np.array(thr), np.array(plug)


def summarize(name, y, p, fold):
    thr, plug = fold_f1(y, p, fold)
    s = {"auc": roc_auc_score(y, p), "logloss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)),
         "f1_oof_best": best_threshold(y, p)[0], "f1_thr_cv": thr.mean(), "f1_plugin_cv": plug.mean()}
    print(f"  {name:8s} auc {s['auc']:.4f}  logloss {s['logloss']:.4f}  F1 best-thr(in-sample) "
          f"{s['f1_oof_best']:.4f}  thr-cv {thr.mean():.4f}+/-{thr.std(ddof=1) / np.sqrt(FOLDS):.4f}  "
          f"plugin-cv {plug.mean():.4f}+/-{plug.std(ddof=1) / np.sqrt(FOLDS):.4f}")
    return s, thr, plug


def run_arms(X, Xte, y, repeats, smoke):
    res = {}
    signal.signal(signal.SIGALRM, on_alarm)
    for name, fn in ARMS.items():
        t0, oof_sum, te_reps = time.time(), np.zeros(len(y)), []
        signal.alarm(max(60, BUDGET[name] // (10 if smoke else 1)))
        try:
            for r in range(repeats):
                oof, te_r = np.full(len(y), np.nan), np.zeros(len(Xte))
                for k, (a, b) in enumerate(StratifiedKFold(FOLDS, shuffle=True, random_state=r).split(X, y)):
                    pv, pt = fn(X.iloc[a], y[a], [X.iloc[b], Xte], seed=100 * r + k, smoke=smoke)
                    oof[b], te_r = pv, te_r + pt / FOLDS
                assert np.isfinite(oof).all() and np.isfinite(te_r).all(), f"{name}: non-finite predictions"
                oof_sum += oof
                te_reps.append(te_r)
                print(f"  {name} repeat {r}: auc {roc_auc_score(y, oof):.4f}  {time.time() - t0:.0f}s")
        except Exception as e:
            print(f"  {name}: stopped after {len(te_reps)} repeat(s) ({type(e).__name__}: {str(e)[:300]})")
        finally:
            signal.alarm(0)
        if te_reps:
            res[name] = (oof_sum / len(te_reps), np.mean(te_reps, axis=0))
        print(f"  {name}: {len(te_reps)} repeat(s), {time.time() - t0:.0f}s")
    assert res, "every family failed"
    return res


def run(X, Xte, y, ids_tr, ids_te, smoke):
    repeats = 1 if smoke else REPEATS
    out = OUT / "extra" / ("smoke" if smoke else VERSION)
    out.mkdir(parents=True, exist_ok=True)
    print(f"=== {'smoke' if smoke else 'full'} run: {len(y)} train rows, {FOLDS} folds x {repeats} repeat(s)")
    res = run_arms(X, Xte, y, repeats, smoke)
    names = list(res)
    fold = np.empty(len(y), dtype=int)
    for k, (_, b) in enumerate(StratifiedKFold(FOLDS, shuffle=True, random_state=0).split(X, y)):
        fold[b] = k

    L = np.column_stack([logit(res[n][0]) for n in names])
    Lte = np.column_stack([logit(res[n][1]) for n in names])
    p_stack_cv = np.empty(len(y))
    for k in range(FOLDS):
        v = fold == k
        p_stack_cv[v] = apply_stack(fit_stack(L[~v], y[~v]), L[v])
    w = fit_stack(L, y)
    p_te = apply_stack(w, Lte)
    print("stack weights:", {n: round(float(x), 3) for n, x in zip(names, w[:-1])}, f"intercept {w[-1]:.3f}")

    print("OOF (averaged over repeats; stack cross-fitted by fold):")
    report, per_fold = {}, {}
    for n in names:
        report[n], thr, _ = summarize(n, y, res[n][0], fold)
        per_fold[n] = thr
    report["stack"], thr_s, plug_s = summarize("stack", y, p_stack_cv, fold)
    best = max(names, key=lambda n: report[n]["f1_thr_cv"])
    d = thr_s - per_fold[best]
    print(f"stack - best single ({best}), thr-cv F1 per fold: {d.mean():+.4f} +/- {d.std(ddof=1) / np.sqrt(FOLDS):.4f}")

    d_rule = thr_s - plug_s
    se_rule = d_rule.std(ddof=1) / np.sqrt(FOLDS)
    use_thr = d_rule.mean() > 2 * se_rule
    f1_star, thr_star = best_threshold(y, p_stack_cv)
    lab_plug = plugin_labels(p_te)
    lab_thr = (p_te > thr_star).astype(int)
    lab = lab_thr if use_thr else lab_plug
    print(f"rule: thr-cv - plugin-cv = {d_rule.mean():+.4f} +/- {se_rule:.4f} -> "
          f"{'OOF threshold' if use_thr else 'plug-in top-k'}")
    print(f"test: mean p {p_te.mean():.4f} (train rate {y.mean():.4f}); positives plug-in {lab_plug.sum()} "
          f"({lab_plug.mean():.3f}), OOF threshold {thr_star:.4f} -> {lab_thr.sum()} ({lab_thr.mean():.3f}); "
          f"labels differing {int((lab_plug != lab_thr).sum())}")

    sub = pd.DataFrame({"Id": ids_te, TARGET: lab})
    assert len(sub) == len(Xte) and sub["Id"].is_unique and sub[TARGET].isin([0, 1]).all()
    alt = pd.DataFrame({"Id": ids_te, TARGET: lab_plug if use_thr else lab_thr})
    pd.DataFrame({"Id": ids_tr, TARGET: y, **{n: res[n][0] for n in names}, "stack_cv": p_stack_cv,
                  "fold": fold}).to_csv(out / "oof.csv", index=False)
    pd.DataFrame({"Id": ids_te, **{n: res[n][1] for n in names}, "stack": p_te}).to_csv(out / "test_pred.csv", index=False)
    alt.to_csv(out / f"submission_{'plugin' if use_thr else 'oofthr'}.csv", index=False)
    report["decision"] = {"rule": "oof_threshold" if use_thr else "plugin_topk", "thr_star": thr_star,
                          "f1_star_in_sample": f1_star, "thr_minus_plugin": float(d_rule.mean()),
                          "thr_minus_plugin_se": float(se_rule), "positives": int(lab.sum()),
                          "weights": {n: float(x) for n, x in zip(names, w[:-1])}, "intercept": float(w[-1])}
    (out / "report.json").write_text(json.dumps(report, indent=1, default=float))
    path = (out if smoke else OUT) / "submission.csv"
    sub.to_csv(path, index=False)
    return path


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    t_start = time.time()
    install_pytabkit()
    tr, te = load()
    X, Xte, _ = features(tr, te)
    y = tr[TARGET].values.astype(int)
    ids_tr, ids_te = tr["Id"].values, te["Id"].values
    s, _ = train_test_split(np.arange(len(y)), train_size=SMOKE_ROWS, stratify=y, random_state=0)
    run(X.iloc[s].reset_index(drop=True), Xte, y[s], ids_tr[s], ids_te, smoke=True)
    print(f"smoke ok ({time.time() - t_start:.0f}s)")
    if not SMOKE_ONLY:
        path = run(X, Xte, y, ids_tr, ids_te, smoke=False)
        print(f"total {time.time() - t_start:.0f}s")
        print(f"SUBMIT: {path}")
