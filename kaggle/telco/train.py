import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

from preprocess import ID, TARGET, TENURE_MAX, Imputer, clean, load, make

VERSION = "V1"
OUT = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path(".")
FULL = dict(repeats=20, folds=5)
SMOKE = dict(repeats=1, folds=2)
LADDER = ["starter", "clean", "onehot", "scale", "log", "spline", "eng", "inter"]
STEP = {"starter": {"starter": True}, "clean": {}, "onehot": {"onehot": True}, "scale": {"scale": True},
        "log": {"log": True}, "spline": {"spline": True}, "eng": {"eng": True}, "inter": {"inter": True}}
SIDE = {"mc_services": {"mc_fill": "services"}, "mc_avg": {"mc_fill": "avg"}, "te": {"te": True}}
USES_LABELS = {"te"}


def fixed_model():
    return LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=42)


def configs():
    cfgs, cur = {"starter": STEP["starter"]}, {}
    for name in LADDER[1:]:
        cur = {**cur, **STEP[name]}
        cfgs[name] = cur
    for name, extra in SIDE.items():
        cfgs[name] = {**cfgs[LADDER[-1]], **extra}
    return cfgs


def fit_predict(X, y, Xt):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", ConvergenceWarning)
        p = fixed_model().fit(X, y).predict_proba(Xt)[:, 1]
    return p, sum(issubclass(x.category, ConvergenceWarning) for x in w)


def imputation_check(tr, te):
    d = clean(pd.concat([tr.drop(columns=TARGET), te], ignore_index=True))
    print("defects before repair (train+test): " + ", ".join(f"{c} {d[c].isna().sum()}" for c in
          ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]))
    imp = Imputer().fit(d)
    k = d.tenure.between(1, TENURE_MAX) & d.MonthlyCharges.notna() & d.TotalCharges.notna()
    t, mc, tc = d.tenure[k], d.MonthlyCharges[k], d.TotalCharges[k]
    est = np.clip(np.round(tc / mc), 1, TENURE_MAX)
    print(f"tenure from round(TC/MC) on {k.sum()} known rows: exact {np.mean(est == t):.3f}, "
          f"|err| <= 1 {np.mean((est - t).abs() <= 1):.3f}, MAE {np.mean((est - t).abs()):.2f} months")
    hat = imp.mc_hat(d[k])
    for name, v in {"ratio TC/tenure": tc / t, "services price list": hat, "avg": (tc / t + hat) / 2}.items():
        e = (v - mc).abs()
        print(f"MonthlyCharges by {name}: MAE {e.mean():.2f}, p90 {e.quantile(0.9):.2f}, p99 {e.quantile(0.99):.2f}")
    print("price list: " + ", ".join(f"{n} {c:.2f}" for n, c in zip(
        ["base", "phone", "multi", "dsl", "fiber", "security", "backup", "protect", "support", "tv", "movies"], imp.price)))


def cross_validate(tr, cfgs, repeats, folds):
    X, y = tr.drop(columns=TARGET), tr[TARGET].to_numpy()
    splits = list(RepeatedStratifiedKFold(n_splits=folds, n_repeats=repeats, random_state=0).split(X, y))
    auc, t0 = {}, time.time()
    for name, cfg in cfgs.items():
        oof, warn = np.zeros((repeats, len(y))), 0
        for s, (i, j) in enumerate(splits):
            Xa, Xb = make(X.iloc[i], X.iloc[j], y[i], cfg)
            oof[s // folds, j], w = fit_predict(Xa, y[i], Xb)
            warn += w
        auc[name] = np.array([roc_auc_score(y, o) for o in oof])
        print(f"  {name:12s} {Xa.shape[1]:3d} cols  AUC {auc[name].mean():.5f} (sd over repeats {auc[name].std():.5f})"
              f"{'  convergence warnings ' + str(warn) if warn else ''}  ({time.time() - t0:.0f}s)", flush=True)
    return auc


def paired(auc, a, b):
    d = auc[a] - auc[b]
    return f"{a} - {b}: {d.mean():+.5f} +- {d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float('nan'):.5f}"


def report(auc):
    print("ladder steps (paired over repeats, +- standard error):")
    for a, b in zip(LADDER[1:], LADDER[:-1]):
        print("  " + paired(auc, a, b))
    print("  " + paired(auc, LADDER[-1], "starter"))
    for name in SIDE:
        print("  " + paired(auc, name, LADDER[-1]) + ("  [uses labels in preprocessing]" if name in USES_LABELS else ""))


def write(tr, te, cfgs, auc):
    X, y = tr.drop(columns=TARGET), tr[TARGET].to_numpy()
    extra = OUT / "extra" / VERSION
    extra.mkdir(parents=True, exist_ok=True)
    subs = {}
    for name, cfg in cfgs.items():
        Xa, Xb = make(X, te, y, cfg)
        p, _ = fit_predict(Xa, y, Xb)
        assert len(p) == len(te) and np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()
        subs[name] = pd.DataFrame({"Id": te[ID], "Expected": p})
        subs[name].to_csv(extra / f"{name}.csv", index=False)
    pool = [n for n in cfgs if n not in USES_LABELS]
    best = max(pool, key=lambda n: auc[n].mean())
    print(f"submission: {best} (CV AUC {auc[best].mean():.5f}); every arm is in {extra}/")
    subs[best].to_csv(OUT / "submission.csv", index=False)
    print(f"SUBMIT: {OUT / 'submission.csv'}")


def run(tr, te, repeats, folds, smoke):
    cfgs = configs()
    print(f"{'smoke' if smoke else 'full'}: {repeats} x {folds}-fold CV, {len(cfgs)} configurations")
    auc = cross_validate(tr, cfgs, repeats, folds)
    if not smoke:
        report(auc)
        write(tr, te, cfgs, auc)


if __name__ == "__main__":
    t0 = time.time()
    tr, te = load()
    imputation_check(tr, te)
    run(tr, te, smoke=True, **SMOKE)
    print("smoke ok")
    run(tr, te, smoke=False, **FULL)
    print(f"total {(time.time() - t0) / 60:.1f} min")
