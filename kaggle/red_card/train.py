import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from preprocess import FEATURES, load

VERSION = "V2.2"
T_START = time.time()
SEEDS = [42, 7, 2026, 11, 99]
REFIT_SCALE = 1.2
LGB_PARAMS = dict(objective="binary", learning_rate=0.02, num_leaves=15, min_child_samples=300, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0, verbose=-1)
N_LGB, SMOKE_LGB = 5000, 30
ARMS = {"lgb": FEATURES, "lgb_v21": [c for c in FEATURES if c != "dow"]}
SUBMIT_ARM = "lgb"
SPLITS = {"near": ("2020-01-01", "2020-01-01", "2022-01-01"), "far": ("2019-01-01", "2021-01-01", "2022-01-01")}
BOOT = 300
OUT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")

def fit_lgb(X, y, seed, n, Xm=None, ym=None):
    m = lgb.LGBMClassifier(**LGB_PARAMS, n_estimators=n, random_state=seed)
    if Xm is None:
        m.fit(X, y)
        return m, n
    m.fit(X, y, eval_set=[(Xm, ym)], callbacks=[lgb.early_stopping(300, verbose=False)])
    return m, m.best_iteration_

def run_arms(a, fa, vb, y, seeds, n, iters=None):

    p, got, models = {}, {}, {}
    for name, cols in ARMS.items():
        X = a[cols]
        p[name], got[name] = np.zeros(vb.sum()), []
        for s in seeds:
            if iters is None:
                m, it = fit_lgb(X[fa], y[fa], s, n, X[vb], y[vb])
            else:
                m, it = fit_lgb(X[fa], y[fa], s, max(1, int(iters[name] * REFIT_SCALE)))
            p[name] += m.predict_proba(X[vb])[:, 1] / len(seeds)
            got[name].append(it)
        models[name] = m
    return p, {k: int(np.median(v)) for k, v in got.items()}, models

def f1_curve(p, y):
    o = np.argsort(-p)
    return 2 * np.cumsum(y[o]) / (np.arange(1, len(y) + 1) + y.sum())

def best_f1(p, y):
    f = f1_curve(p, y)
    i = int(np.argmax(f))
    return (i + 1) / len(y), float(f[i])

def f1_at(p, y, t):
    c = p > t
    return 2 * (c & (y == 1)).sum() / (c.sum() + y.sum())

def summary(name, p, y, years, starts):
    q, f = best_f1(p, y)
    curve = f1_curve(p, y)
    at = " ".join(f"{s:.0%}:{curve[int(s * len(y)) - 1]:.4f}" for s in (0.03, 0.04, 0.05, 0.06))
    by_year = " ".join(f"{yr}:{best_f1(p[years == yr], y[years == yr])[1]:.4f}" for yr in np.unique(years))
    print(f"  {name:8s} AUC {roc_auc_score(y, p):.4f} logloss {log_loss(y, p):.5f} mean p {p.mean():.4f} | best F1 {f:.4f} at top {q:.2%}, F1/2 rule {f1_at(p, y, f / 2):.4f} | {at} | {by_year}", flush=True)
    b = pd.cut(starts, [-1, 30, 60, 90, 120, 400])
    cal = pd.DataFrame({"b": b, "y": y, "p": p}).groupby("b", observed=True).agg(n=("y", "size"), rate=("y", "mean"), p=("p", "mean"))
    print("    calibration by starts: " + " ".join(f"{i}: n {r.n} rate {r.rate:.4f} p {r.p:.4f}" for i, r in cal.iterrows()))

def paired(p, y, groups, ref):

    rng = np.random.default_rng(0)
    idx = pd.Series(np.arange(len(y))).groupby(groups).indices
    keys = list(idx)
    loss = {k: -(y * np.log(np.clip(v, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - v, 1e-6, 1))) for k, v in p.items()}
    d = {k: [] for k in p if k != ref}
    for _ in range(BOOT):
        b = np.concatenate([idx[keys[i]] for i in rng.integers(0, len(keys), len(keys))])
        fr = f1_curve(p[ref][b], y[b])[int(0.045 * len(b)) - 1]
        for k in d:
            d[k].append((loss[k][b].mean() - loss[ref][b].mean(), f1_curve(p[k][b], y[b])[int(0.045 * len(b)) - 1] - fr))
    for k, v in d.items():
        v = np.array(v)
        print(f"  {k:8s} vs {ref}: logloss {v[:, 0].mean() * 1e4:+.2f}e-4 ± {v[:, 0].std() * 1e4:.2f}e-4, F1@4.5% {v[:, 1].mean():+.4f} ± {v[:, 1].std():.4f}")

def write(pt, t, a, test_ids, extra):
    te = a.y.isna().to_numpy()
    sub = pd.DataFrame({"player_match_id": a.player_match_id[te].to_numpy(), "red_card": (pt > t).astype(int)})
    sub = sub.set_index("player_match_id").loc[test_ids].reset_index()
    assert len(sub) == 105512 and sub.player_match_id.is_unique and sub.red_card.isin([0, 1]).all()
    path = OUT / "submission.csv"
    sub.to_csv(path, index=False)
    d = OUT / "extra" / VERSION
    d.mkdir(parents=True, exist_ok=True)
    for name, df in extra.items():
        df.to_csv(d / f"{name}.csv", index=False)
    return path, int(sub.red_card.sum())

if __name__ == "__main__":
    a, test_ids = load()
    lab = a.y.notna().to_numpy()
    te = ~lab
    y, yr, starts = a.y.to_numpy(), a.k.dt.year.to_numpy(), a.starts.to_numpy()
    print(f"load {time.time() - T_START:.0f}s; dow counts {a.dow.value_counts().to_dict()}", flush=True)
    sm, _, _ = run_arms(a, lab, te, y, SEEDS[:1], SMOKE_LGB, {k: SMOKE_LGB for k in ARMS})
    write(sm[SUBMIT_ARM], 0.05, a, test_ids, {})
    print(f"smoke ok ({time.time() - T_START:.0f}s)", flush=True)
    res = {}
    for split, (fit_end, val_start, val_end) in SPLITS.items():
        fa = lab & (a.k < pd.Timestamp(fit_end, tz="UTC")).to_numpy()
        vb = lab & (a.k >= pd.Timestamp(val_start, tz="UTC")).to_numpy() & (a.k < pd.Timestamp(val_end, tz="UTC")).to_numpy()
        p, iters, models = run_arms(a, fa, vb, y, SEEDS, N_LGB)
        print(f"== {split}: fit {fa.sum()} rows to {fit_end}, val {vb.sum()} rows (red rate {y[vb].mean():.4f}), iters {iters} ({time.time() - T_START:.0f}s)", flush=True)
        for name, pv in p.items():
            summary(name, pv, y[vb], yr[vb], starts[vb])
        paired(p, y[vb], a.match_id.to_numpy()[vb], "lgb_v21")
        res[split] = (p, iters, vb)
        if split == "near":
            m = models[SUBMIT_ARM]
            imp = pd.Series(m.booster_.feature_importance("gain"), index=ARMS[SUBMIT_ARM]).sort_values(ascending=False)
            print("  lgb gain: " + ", ".join(f"{c} {v / imp.sum():.3f}" for c, v in imp.items()))
    pv, iters, vb = res["near"]
    _, f = best_f1(pv[SUBMIT_ARM], y[vb])
    t = f / 2
    print(f"threshold = near-val best F1 / 2 = {t:.4f}; val share {np.mean(pv[SUBMIT_ARM] > t):.2%}")
    pt, _, _ = run_arms(a, lab, te, y, SEEDS, N_LGB, iters)
    for name, v in pt.items():
        print(f"  test {name:8s} mean p by year: " + " ".join(f"{yy}:{v[yr[te] == yy].mean():.4f}" for yy in np.unique(yr[te])))
    call = pt[SUBMIT_ARM] > t
    print("test share called red by year: " + " ".join(f"{yy}:{call[yr[te] == yy].mean():.2%}" for yy in np.unique(yr[te])))
    extra = {"val_pred": pd.DataFrame({"player_match_id": a.player_match_id[vb].to_numpy(), "y": y[vb], **pv}),
             "test_pred": pd.DataFrame({"player_match_id": a.player_match_id[te].to_numpy(), **pt})}
    path, n1 = write(pt[SUBMIT_ARM], t, a, test_ids, extra)
    print(f"called red: {n1} of {te.sum()} ({n1 / te.sum():.2%})")
    print(f"total {(time.time() - T_START) / 60:.1f} min")
    print(f"SUBMIT: {path}")
