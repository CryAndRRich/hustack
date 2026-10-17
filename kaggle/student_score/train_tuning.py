import json
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from preprocess import TARGET, get_out_dir, load_data
from tune import GAM_BASE, LGB_BASE, fit_gam, fit_lgb, tune

VERSION = "v1_2"
AUTO_SMOKE = True
CFG = {
    "outer_folds": 5, "inner_folds": 5, "final_lgb_folds": 5,
    "max_rounds": 5000, "early_stop": 200,
    "gam_trials": 200, "gam_timeout": 150,
    "lgb_trials": 100, "lgb_timeout": 240,
}
SMOKE_CFG = {
    "outer_folds": 2, "inner_folds": 2, "final_lgb_folds": 2,
    "max_rounds": 50, "early_stop": 10,
    "gam_trials": 3, "gam_timeout": 30,
    "lgb_trials": 3, "lgb_timeout": 30,
}
SEED = 42
GATE_T = -2.0
ARMS = ["gam_base", "gam_tuned", "lgb_base", "lgb_tuned"]


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def paired(y, a, b):
    d = (y - a) ** 2 - (y - b) ** 2
    mean, se = d.mean(), d.std(ddof=1) / np.sqrt(len(d))
    return mean, se, mean / se


def fit_predict(arm, p, X_tr, y_tr, X_out, cfg, seed):
    if arm.startswith("gam"):
        return fit_gam(p, X_tr, y_tr).predict(X_out)
    return fit_lgb(p, X_tr, y_tr, cfg, seed).predict(X_out)


def nested(X, y, cfg):
    oof = {a: np.zeros(len(y)) for a in ARMS}
    chosen = {"gam": [], "lgb": []}
    for k, (tr, va) in enumerate(KFold(cfg["outer_folds"], shuffle=True, random_state=SEED).split(X)):
        t0 = time.time()
        print(f"  outer fold {k}: tuning on {len(tr)} rows")
        params = {"gam_base": GAM_BASE, "lgb_base": LGB_BASE}
        for fam in ("gam", "lgb"):
            best, _ = tune(fam, X.iloc[tr], y[tr], cfg, SEED + 1000 + k)
            params[f"{fam}_tuned"] = best
            chosen[fam].append(best)
        for arm in ARMS:
            oof[arm][va] = fit_predict(arm, params[arm], X.iloc[tr], y[tr], X.iloc[va], cfg, SEED + k)
        line = "  ".join(f"{a} {rmse(y[va], oof[a][va]):.4f}" for a in ARMS)
        print(f"  outer fold {k}: {line}  ({time.time() - t0:.0f}s)")
    return oof, chosen


def final_predictions(X, y, X_test, cfg):
    print("  final tuning on all rows")
    params = {"gam_base": GAM_BASE, "lgb_base": LGB_BASE}
    for fam in ("gam", "lgb"):
        params[f"{fam}_tuned"], _ = tune(fam, X, y, cfg, SEED + 2000)
    test = {}
    for arm in ARMS:
        if arm.startswith("gam"):
            test[arm] = fit_gam(params[arm], X, y).predict(X_test)
        else:
            preds = [fit_lgb(params[arm], X.iloc[tr], y[tr], cfg, SEED + k).predict(X_test)
                     for k, (tr, _) in enumerate(KFold(cfg["final_lgb_folds"], shuffle=True,
                                                       random_state=SEED).split(X))]
            test[arm] = np.mean(preds, axis=0)
    return test, params


def nested_blend(y, p1, p2, n_folds=5):
    out = np.zeros(len(y))
    for tr, va in KFold(n_folds, shuffle=True, random_state=SEED).split(p1):
        diff = p2[tr] - p1[tr]
        w = float(np.clip(np.dot(diff, y[tr] - p1[tr]) / np.dot(diff, diff), 0, 1))
        out[va] = p1[va] + w * (p2[va] - p1[va])
    diff = p2 - p1
    return out, float(np.clip(np.dot(diff, y - p1) / np.dot(diff, diff), 0, 1))


def report(y, oof, chosen):
    print("\n=== nested outer CV (tuning never saw the scored fold) ===")
    for a in ARMS:
        print(f"  {a:10s} {rmse(y, oof[a]):.4f}")
    for a, b in [("gam_tuned", "gam_base"), ("lgb_tuned", "lgb_base"), ("lgb_tuned", "gam_tuned"),
                 ("gam_base", "lgb_base")]:
        m, se, t = paired(y, oof[a], oof[b])
        print(f"  paired MSE {a} - {b}: {m:+.4f} ± {se:.4f} (t = {t:+.2f})")
    print(f"  corr(gam_tuned, lgb_tuned) OOF: {np.corrcoef(oof['gam_tuned'], oof['lgb_tuned'])[0, 1]:.4f}")
    for fam, plist in chosen.items():
        print(f"\n  {fam} params chosen per outer fold:")
        print(pd.DataFrame(plist).round(4).to_string())

    pick, pick_oof = "gam_base", oof["gam_base"]
    m, se, t = paired(y, oof["gam_tuned"], pick_oof)
    if t < GATE_T:
        pick, pick_oof = "gam_tuned", oof["gam_tuned"]
    blend_oof, w_full = nested_blend(y, oof["gam_tuned"], oof["lgb_tuned"])
    m, se, t = paired(y, blend_oof, pick_oof)
    print(f"\n  nested blend gam_tuned+lgb_tuned: RMSE {rmse(y, blend_oof):.4f}  lgb weight {w_full:.3f}  "
          f"vs {pick}: {m:+.4f} ± {se:.4f} (t = {t:+.2f})")
    if t < GATE_T:
        pick = "blend"
    print(f"  gate (t < {GATE_T}) picks: {pick}")
    return pick, w_full


def write_outputs(y, oof, test, params, pick, w_full, train_ids, sample):
    out = get_out_dir()
    test = dict(test)
    test["blend"] = test["gam_tuned"] + w_full * (test["lgb_tuned"] - test["gam_tuned"])
    for name, p in test.items():
        p = np.clip(p, 0, 100)
        assert len(p) == len(sample) and np.isfinite(p).all(), name
        sub = pd.DataFrame({"id": sample["id"].to_numpy(), TARGET: p})
        assert list(sub.columns) == list(sample.columns)
        sub.to_csv(out / f"submission_{name}.csv", index=False)
        if name == pick:
            sub.to_csv(out / "submission.csv", index=False)
        print(f"  submission_{name}.csv  mean {p.mean():.3f}  sd {p.std():.3f}  min {p.min():.2f}  max {p.max():.2f}")
    print(f"  submission.csv = submission_{pick}.csv")
    pd.DataFrame({"id": train_ids, TARGET: y, **{f"oof_{a}": v for a, v in oof.items()}}).to_csv(
        out / f"oof_{VERSION}.csv", index=False)
    with open(out / f"params_{VERSION}.json", "w") as f:
        json.dump({k: {kk: (vv.item() if hasattr(vv, "item") else vv) for kk, vv in v.items()}
                   for k, v in params.items()}, f, indent=1)
    print(f"  wrote oof_{VERSION}.csv, params_{VERSION}.json")
    print("  final params:\n" + json.dumps({k: params[k] for k in ("gam_tuned", "lgb_tuned")}, indent=1, default=str))


def run(X, y, X_test, train_ids, sample, cfg):
    oof, chosen = nested(X, y, cfg)
    pick, w_full = report(y, oof, chosen)
    test, params = final_predictions(X, y, X_test, cfg)
    write_outputs(y, oof, test, params, pick, w_full, train_ids, sample)


def main():
    X, y, X_test, train_ids, sample = load_data()
    if AUTO_SMOKE:
        t0 = time.time()
        print("\n=== smoke pass (result discarded) ===")
        run(X, y, X_test, train_ids, sample, SMOKE_CFG)
        print(f"smoke ok in {time.time() - t0:.0f}s")
    t0 = time.time()
    print(f"\n=== {VERSION}: Optuna-tuned GAM and LightGBM, nested {CFG['outer_folds']}-fold ===")
    run(X, y, X_test, train_ids, sample, CFG)
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
