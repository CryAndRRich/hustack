import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.stats import rankdata
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BANDWIDTH = 30
REPEATS = 3
OUT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")


def data_dir():
    for p in Path("/kaggle/input").rglob("train.csv") if Path("/kaggle/input").exists() else []:
        if "589" in pd.read_csv(p, nrows=1).columns:
            return p.parent
    try:
        return Path(__file__).resolve().parent.parent / "data"
    except NameError:
        return Path.cwd() / "data"


def load():
    d = data_dir()
    tr, te = pd.read_csv(d / "train.csv"), pd.read_csv(d / "test.csv")
    assert set(tr.target.unique()) == {-1, 1}
    y = (tr.target.values == 1).astype(int)
    feats = [c for c in te.columns if c not in ("id", "time") and tr[c].nunique() > 1]
    return tr, te, y, feats


def neighbour_rate(ids_fit, y_fit, ids_query):
    w = np.exp(-0.5 * ((ids_query[:, None] - ids_fit[None, :]) / BANDWIDTH) ** 2)
    w[ids_query[:, None] == ids_fit[None, :]] = 0.0
    return (w @ y_fit + y_fit.mean()) / (w.sum(1) + 1.0)


def design(fit, query, y_fit, feats):
    med = fit[feats].median()
    rate = lambda q: neighbour_rate(fit.id.values, y_fit, q.id.values)[:, None]
    return (np.hstack([fit[feats].fillna(med).values, rate(fit)]),
            np.hstack([query[feats].fillna(med).values, rate(query)]))


FAMILIES = {
    "logistic": lambda: lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.005, max_iter=2000)),
    "lightgbm": lambda: lambda: LGBMClassifier(n_estimators=400, learning_rate=0.02, num_leaves=7,
                                               min_child_samples=20, colsample_bytree=0.3, subsample=0.8,
                                               subsample_freq=1, reg_lambda=5.0, verbose=-1, random_state=0),
    "extratrees": lambda: lambda: ExtraTreesClassifier(n_estimators=1000, min_samples_leaf=3,
                                                       max_features="sqrt", n_jobs=-1, random_state=0),
}


def run(tr, te, y, feats):
    splits = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=REPEATS, random_state=0).split(tr, y))
    folds = [design(tr.iloc[a], tr.iloc[b], y[a], feats) for a, b in splits]
    full = design(tr, te, y, feats)
    oof, test = {}, {}
    for name, factory in FAMILIES.items():
        t0 = time.time()
        try:
            make = factory()
            oof[name] = [make().fit(Xa, y[a]).predict_proba(Xb)[:, 1] for (a, _), (Xa, Xb) in zip(splits, folds)]
            test[name] = make().fit(full[0], y).predict_proba(full[1])[:, 1]
        except Exception as e:
            print(f"{name:10s} FAILED: {e!r}")
            oof.pop(name, None)
            continue
        print(f"{name:10s} {time.time() - t0:6.0f} s")
    oof["blend"] = [np.mean([rankdata(oof[n][i]) for n in test], 0) for i in range(len(splits))]
    test["blend"] = np.mean([rankdata(test[n]) for n in test], 0)
    scores = {}
    for name, preds in oof.items():
        aucs = [roc_auc_score(y[b], p) for (_, b), p in zip(splits, preds)]
        scores[name] = np.mean(aucs)
        print(f"{name:10s} cv auc {np.mean(aucs):.4f} ± {np.std(aucs) / np.sqrt(len(aucs)):.4f}")
    return scores, test


def write(te, scores, test):
    best = max(scores, key=scores.get)
    norm = lambda p: (rankdata(p) - 1) / (len(p) - 1)
    pd.DataFrame({"id": te.id, "target": norm(test[best])}).to_csv(OUT / "submission.csv", index=False)
    with zipfile.ZipFile(OUT / "extra.zip", "w") as z:
        for name in test:
            if name != best:
                z.writestr(f"{name}.csv", pd.DataFrame({"id": te.id, "target": norm(test[name])}).to_csv(index=False))
    print(f"SUBMIT: submission.csv  ({best}, cv auc {scores[best]:.4f})")


if __name__ == "__main__":
    tr, te, y, feats = load()
    print(f"{len(feats)} non-constant sensors + neighbour rate, {y.sum()} fails of {len(y)}")
    scores, test = run(tr, te, y, feats)
    write(te, scores, test)
