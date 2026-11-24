import importlib
import shutil
import subprocess
import sys
import traceback
import zipfile

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from preprocess import ID, LINES, OUT, TGT, WORK, load, say

FOLDS, REPEATS, SEED = 5, 3, 0
MEMBER_TOL = 0.01
SMOKE_FIT, SMOKE_PRED = 1000, 300
PACKAGES = {"pytabkit": "pytabkit", "interpret-core": "interpret.glassbox"}
TD_MODELS = {"lgb_td": "LGBM_TD_Classifier", "xgb_td": "XGB_TD_Classifier",
             "cb_td": "CatBoost_TD_Classifier", "realmlp_td": "RealMLP_TD_Classifier"}


def ensure(pkg, module):
    try:
        importlib.import_module(module)
        return True
    except ImportError:
        pass
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], capture_output=True, text=True)
    if r.returncode:
        say("pip install failed", pkg, r.stderr[-800:])
        return False
    importlib.invalidate_caches()
    try:
        importlib.import_module(module)
        return True
    except Exception:
        say("import failed after install", module, traceback.format_exc()[-800:])
        return False


def lr(a, y, b):
    m = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=10000))
    return m.fit(a, y).predict_proba(b)[:, 1]


def spline_lr(spline_cols):
    def fit_predict(a, y, b):
        ct = ColumnTransformer([("spl", SplineTransformer(n_knots=5, degree=3, knots="uniform"), spline_cols)],
                               remainder="passthrough")
        m = make_pipeline(ct, StandardScaler(), LogisticRegression(C=0.03, max_iter=10000))
        return m.fit(a, y).predict_proba(b)[:, 1]
    return fit_predict


def lgb_explore(a, y, b):
    m = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.02, num_leaves=15, min_child_samples=30, subsample=0.8,
                           subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0, random_state=SEED, verbose=-1)
    return m.fit(a, y).predict_proba(b)[:, 1]


def ebm(a, y, b):
    from interpret.glassbox import ExplainableBoostingClassifier
    to_str = lambda d: d.assign(**{c: d[c].astype(str) for c in d.columns if d[c].dtype.name == "category"})
    return ExplainableBoostingClassifier(random_state=SEED).fit(to_str(a), y).predict_proba(to_str(b))[:, 1]


def td(cls_name):
    def fit_predict(a, y, b):
        cls = getattr(importlib.import_module("pytabkit"), cls_name)
        return np.asarray(cls(random_state=SEED).fit(a, y).predict_proba(b))[:, 1]
    return fit_predict


def rows(X, idx):
    return X.iloc[idx] if isinstance(X, pd.DataFrame) else X[idx]


def smoke(name, fn, X, y):
    p = np.asarray(fn(rows(X, np.arange(SMOKE_FIT)), y[:SMOKE_FIT], rows(X, np.arange(SMOKE_FIT, SMOKE_FIT + SMOKE_PRED))), float)
    assert p.shape == (SMOKE_PRED,) and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), p[:5]
    say(f"smoke ok {name}: AUC on {SMOKE_PRED} rows {roc_auc_score(y[SMOKE_FIT:SMOKE_FIT + SMOKE_PRED], p):.3f}")


def run_model(fn, X, y, n, splits):
    Xtr, Xte = rows(X, np.arange(n)), rows(X, np.arange(n, len(X)))
    oof = np.zeros((REPEATS, n))
    for k, (f, v) in enumerate(splits):
        oof[k // FOLDS, v] = fn(rows(Xtr, f), y[f], rows(Xtr, v))
    test = np.asarray(fn(Xtr, y, Xte), float)
    assert np.isfinite(oof).all() and np.isfinite(test).all() and len(test) == len(X) - n
    return oof, test


def fold_aucs(y, oof, splits):
    return np.array([roc_auc_score(y[v], oof[k // FOLDS, v]) for k, (_, v) in enumerate(splits)])


def rank_mean(preds):
    return np.mean([rankdata(p, axis=-1) / p.shape[-1] for p in preds], axis=0)


def main():
    shutil.rmtree(WORK, ignore_errors=True)
    (WORK / "extra").mkdir(parents=True)
    data = load()
    y, n = data["y"], data["n"]
    splits = list(RepeatedStratifiedKFold(n_splits=FOLDS, n_repeats=REPEATS, random_state=SEED).split(np.zeros(n), y))
    ok = {pkg: ensure(pkg, mod) for pkg, mod in PACKAGES.items()}
    say("packages", ok)

    candidates = [("lgb_explore", lgb_explore, "frame"), ("lr", lr, "lin"),
                  ("spline_lr", spline_lr(data["spline_cols"]), "lin")]
    if ok["interpret-core"]:
        candidates.append(("ebm", ebm, "frame"))
    if ok["pytabkit"]:
        candidates += [(name, td(cls), "dense" if name == "realmlp_td" else "frame") for name, cls in TD_MODELS.items()]

    oofs, tests = {}, {}
    for name, fn, key in candidates:
        try:
            smoke(name, fn, data[key], y)
            oofs[name], tests[name] = run_model(fn, data[key], y, n, splits)
            f = fold_aucs(y, oofs[name], splits)
            say(f"{name:12s} mean fold AUC {f.mean():.4f} (sd {f.std(ddof=1):.4f})")
        except Exception:
            say(f"DROPPED {name}:\n{traceback.format_exc()[-1500:]}")
    assert "lgb_explore" in oofs, "the reference model failed"

    aucs = {k: fold_aucs(y, p, splits) for k, p in oofs.items()}
    best = max(aucs, key=lambda k: aucs[k].mean())
    members = [k for k in aucs if (aucs[k] - aucs[best]).mean() >= -MEMBER_TOL]
    oofs["blend"] = rank_mean([oofs[k] for k in members])
    tests["blend"] = rank_mean([tests[k] for k in members])
    aucs["blend"] = fold_aucs(y, oofs["blend"], splits)
    ref = aucs["lgb_explore"]
    table = pd.DataFrame([{"model": k, "auc": a.mean(), "fold_sd": a.std(ddof=1),
                           "d_vs_explore": (a - ref).mean(), "d_sd": (a - ref).std(ddof=1),
                           "d_vs_best": (a - aucs[best]).mean(), "d_best_sd": (a - aucs[best]).std(ddof=1)}
                          for k, a in aucs.items()]).sort_values("auc", ascending=False)
    table.to_csv(WORK / "cv.csv", index=False)
    say("blend members", members)
    say(f"CV table ({REPEATS}x{FOLDS} folds, paired deltas):\n" + table.round(4).to_string(index=False))

    pick = "blend" if len(members) > 1 and (aucs["blend"] - aucs[best]).mean() > 0 else best
    say(f"pick rule: blend if >1 member and it beats the best single ({best}) on mean paired delta; picked {pick}")

    pd.DataFrame({f"{k}_r{r}": p[r] for k, p in oofs.items() for r in range(REPEATS)}).assign(**{TGT: y}).to_csv(
        WORK / "oof.csv", index=False)
    for k, p in tests.items():
        pd.DataFrame({ID: data["test_id"], TGT: np.clip(p, 0, 1)}).to_csv(WORK / "extra" / f"{k}.csv", index=False)
    sub = pd.DataFrame({ID: data["test_id"], TGT: np.clip(tests[pick], 0, 1)})
    assert len(sub) == len(data["test_id"]) and sub[ID].is_unique and sub[TGT].notna().all()
    sub.to_csv(OUT / "submission.csv", index=False)
    say("test rank correlation:\n" + pd.DataFrame({k: rankdata(p) for k, p in tests.items()}).corr().round(3).to_string())
    say(f"SUBMIT: {OUT / 'submission.csv'} ({pick})")

    (WORK / "log.txt").write_text("\n".join(LINES) + "\n")
    with zipfile.ZipFile(OUT / "results.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(WORK.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(WORK))
    for p in OUT.iterdir():
        if p.name not in ("submission.csv", "results.zip", "__notebook__.ipynb"):
            shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink()
    say("outputs:", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
