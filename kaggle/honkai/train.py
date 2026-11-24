import importlib
import shutil
import subprocess
import sys
import traceback
import zipfile

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from preprocess import ID, LINES, OUT, TGT, WORK, load, say

FOLDS, SEED = 5, 0
LR_CS = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
MEMBER_TOL = 0.01
BLEND_TOL = 0.002
SMOKE_FIT, SMOKE_PRED = 800, 200
PACKAGES = {"tabicl": "tabicl", "tabpfn": "tabpfn", "interpret-core": "interpret.glassbox"}


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


def lr_model(C):
    def fit_predict(a, y, b):
        m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=10000))
        return m.fit(a, y).predict_proba(b)[:, 1]
    return fit_predict


def ebm(a, y, b):
    from interpret.glassbox import ExplainableBoostingClassifier
    return ExplainableBoostingClassifier(random_state=SEED, n_jobs=-1).fit(a, y).predict_proba(b)[:, 1]


def tabicl(a, y, b):
    from tabicl import TabICLClassifier
    return TabICLClassifier().fit(a, y).predict_proba(b)[:, 1]


def tabpfn(a, y, b):
    from tabpfn import TabPFNClassifier
    return TabPFNClassifier().fit(a, y).predict_proba(b)[:, 1]


def rows(X, idx):
    return X.iloc[idx] if isinstance(X, pd.DataFrame) else X[idx]


def run_model(fn, X, data, folds):
    tr_idx, te_idx = np.flatnonzero(data["is_tr"]), np.flatnonzero(~data["is_tr"])
    Xtr, Xte, y = rows(X, tr_idx), rows(X, te_idx), data["y"]
    oof = np.zeros(len(y))
    for fit_idx, val_idx in folds:
        oof[val_idx] = fn(rows(Xtr, fit_idx), y[fit_idx], rows(Xtr, val_idx))
    test = fn(Xtr, y, Xte)
    assert np.isfinite(oof).all() and np.isfinite(test).all() and len(test) == len(te_idx)
    return oof, test


def smoke(name, fn, X, data):
    Xtr = rows(X, np.flatnonzero(data["is_tr"]))
    p = fn(rows(Xtr, np.arange(SMOKE_FIT)), data["y"][:SMOKE_FIT], rows(Xtr, np.arange(SMOKE_FIT, SMOKE_FIT + SMOKE_PRED)))
    p = np.asarray(p, float)
    assert p.shape == (SMOKE_PRED,) and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), p[:5]
    say(f"smoke ok {name}: AUC on {SMOKE_PRED} rows {roc_auc_score(data['y'][SMOKE_FIT:SMOKE_FIT + SMOKE_PRED], p):.3f}")


def fold_aucs(y, p, folds):
    return np.array([roc_auc_score(y[v], p[v]) for _, v in folds])


def rank_mean(preds):
    return np.mean([rankdata(p) / len(p) for p in preds], axis=0)


def pack(path):
    (WORK / "log.txt").write_text("\n".join(LINES) + "\n")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(WORK.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(WORK))


def main():
    shutil.rmtree(WORK, ignore_errors=True)
    (WORK / "extra").mkdir(parents=True)
    data = load()
    y = data["y"]
    folds = list(StratifiedGroupKFold(FOLDS, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y, data["groups"]))
    ok = {pkg: ensure(pkg, mod) for pkg, mod in PACKAGES.items()}
    say("packages", ok)

    candidates = [(f"lr_C{C}", lr_model(C), "X_base") for C in LR_CS]
    if ok["interpret-core"]:
        candidates.append(("ebm", ebm, "X_frame"))
    if ok["tabicl"]:
        candidates.append(("tabicl", tabicl, "X_base"))
    if ok["tabpfn"]:
        candidates.append(("tabpfn", tabpfn, "X_base"))

    alive = []
    for name, fn, key in candidates:
        try:
            smoke(name, fn, data[key], data)
            alive.append((name, fn, key))
        except Exception:
            say(f"DROPPED {name} at smoke:\n{traceback.format_exc()[-1500:]}")

    oofs, tests = {}, {}

    def evaluate(name, fn, key):
        try:
            oofs[name], tests[name] = run_model(fn, data[key], data, folds)
            say(f"{name:14s} pooled {roc_auc_score(y, oofs[name]):.4f} folds {np.round(fold_aucs(y, oofs[name], folds), 4)}")
        except Exception:
            say(f"FAILED {name}:\n{traceback.format_exc()[-1500:]}")

    for name, fn, key in alive:
        if name.startswith("lr_"):
            evaluate(name, fn, key)
    lr_names = [n for n in oofs if n.startswith("lr_")]
    best_c_name = max(lr_names, key=lambda n: roc_auc_score(y, oofs[n]))
    best_c = float(best_c_name.removeprefix("lr_C"))
    evaluate("lr_plus", lr_model(best_c), "X_plus")
    lr_best = max([best_c_name, "lr_plus"] if "lr_plus" in oofs else [best_c_name], key=lambda n: roc_auc_score(y, oofs[n]))
    say("best LR", lr_best)
    for name, fn, key in alive:
        if not name.startswith("lr_"):
            evaluate(name, fn, key)

    ref = fold_aucs(y, oofs[lr_best], folds)
    table = []
    for name in oofs:
        f = fold_aucs(y, oofs[name], folds)
        d = f - ref
        table.append({"model": name, "auc_pooled": roc_auc_score(y, oofs[name]), "auc_mean": f.mean(),
                      "d_vs_lr_best": d.mean(), "d_sd": d.std(ddof=1)})
    members = [lr_best] + [n for n in ("ebm", "tabicl", "tabpfn")
                           if n in oofs and roc_auc_score(y, oofs[n]) >= roc_auc_score(y, oofs[lr_best]) - MEMBER_TOL]
    blend_oof = rank_mean([oofs[n] for n in members])
    blend_test = rank_mean([tests[n] for n in members])
    fb = fold_aucs(y, blend_oof, folds)
    table.append({"model": "blend", "auc_pooled": roc_auc_score(y, blend_oof), "auc_mean": fb.mean(),
                  "d_vs_lr_best": (fb - ref).mean(), "d_sd": (fb - ref).std(ddof=1)})
    cv = pd.DataFrame(table).sort_values("auc_pooled", ascending=False)
    cv.to_csv(WORK / "cv.csv", index=False)
    say("blend members", members)
    say("CV table (paired fold deltas vs best LR):\n" + cv.round(4).to_string(index=False))

    use_blend = len(members) > 1 and (fb - ref).mean() >= -BLEND_TOL
    pick = "blend" if use_blend else lr_best
    say(f"pick rule: blend if >1 member and mean paired delta >= -{BLEND_TOL}; picked {pick}")

    tests["blend"] = blend_test
    oofs["blend"] = blend_oof
    pd.DataFrame(oofs).assign(y=y).to_csv(WORK / "oof.csv", index=False)
    for name, p in tests.items():
        pd.DataFrame({ID: data["test_id"], TGT: np.clip(p, 1e-6, 1 - 1e-6)}).to_csv(WORK / "extra" / f"{name}.csv", index=False)
    sub = pd.DataFrame({ID: data["test_id"], TGT: np.clip(tests[pick], 1e-6, 1 - 1e-6)})
    assert len(sub) == 2500 and sub[ID].is_unique and sub[TGT].between(0, 1).all()
    sub.to_csv(OUT / "submission.csv", index=False)
    rho = pd.DataFrame({n: rankdata(p) for n, p in tests.items()}).corr()
    say("test rank correlation:\n" + rho.round(3).to_string())
    say(f"SUBMIT: {OUT / 'submission.csv'} ({pick})")
    pack(OUT / "results.zip")
    for p in OUT.iterdir():
        if p.name not in ("submission.csv", "results.zip", "__notebook__.ipynb"):
            shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink()
    say("outputs:", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
