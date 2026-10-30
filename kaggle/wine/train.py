import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold, train_test_split

from preprocess import FEATURES, ID, LABELS, TARGET, load

VERSION = "v1"
SEED = 42
FULL = {"name": "full", "folds": 5, "repeats": 3, "rows": None, "trees": 4000}
SMOKE = {"name": "smoke", "folds": 2, "repeats": 1, "rows": 3000, "trees": 60}
EXT_WEIGHT = 0.3
GATE_SE = 2.0
PUBLIC_ROWS = (2000, 3000, 5000)
OUT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
LGBM = {"lgbm_shallow": {"max_depth": 3, "num_leaves": 7, "min_child_samples": 100},
        "lgbm_deep": {"max_depth": 5, "num_leaves": 24, "min_child_samples": 50}}
ARMS = ["alcohol_step", "gam", "lgbm_shallow", "lgbm_deep", "catboost_reg", "catboost_cls", "gam_ext",
        "lgbm_shallow_ext"]
REFERENCE = "alcohol_step"


def qwk(y, p):
    n = len(LABELS)
    o = np.bincount((y - LABELS[0]) * n + (p - LABELS[0]), minlength=n * n).reshape(n, n).astype(float)
    e = np.outer(o.sum(1), o.sum(0)) / o.sum()
    w = np.subtract.outer(np.arange(n), np.arange(n)) ** 2 / (n - 1) ** 2
    return 1.0 - (w * o).sum() / (w * e).sum()


def percentile(s):
    return rankdata(s) / len(s)


def to_labels(u, cuts):
    return np.searchsorted(np.sort(cuts), u, side="right") + LABELS[0]


def fit_cuts(y, u, rng):
    base = np.cumsum(np.bincount(y - LABELS[0], minlength=len(LABELS)))[:-1] / len(y)

    def score(c):
        return qwk(y, to_labels(u, np.clip(np.sort(c), 0.0, 1.0)))

    best, best_s = base, score(base)
    for start in [base] + [np.sort(np.clip(base + rng.normal(0, 0.05, base.size), 0, 1)) for _ in range(5)]:
        r = minimize(lambda c: -score(c), start, method="Nelder-Mead",
                     options={"maxiter": 600, "xatol": 1e-4, "fatol": 1e-7})
        c = np.clip(np.sort(r.x), 0.0, 1.0)
        s = score(c)
        if s > best_s:
            best, best_s = c, s
    for step in (0.02, 0.005, 0.001):
        improved = True
        while improved:
            improved = False
            for j in range(best.size):
                for d in (-step, step):
                    c = best.copy()
                    c[j] = np.clip(c[j] + d, 0.0, 1.0)
                    c.sort()
                    s = score(c)
                    if s > best_s + 1e-12:
                        best, best_s, improved = c, s, True
    return best



def alcohol_step(X, y, w, outs, cfg, seed, k=20):
    s = pd.DataFrame({"a": X["alcohol"].to_numpy(), "y": y}).groupby("a")["y"].agg(["sum", "count"])
    m = (s["sum"] + k * y.mean()) / (s["count"] + k)
    return [np.interp(o["alcohol"].to_numpy(), m.index.to_numpy(), m.to_numpy()) for o in outs]


def gam(X, y, w, outs, cfg, seed):
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import RidgeCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import QuantileTransformer, SplineTransformer
    splines = make_pipeline(QuantileTransformer(n_quantiles=1000, random_state=seed),
                            SplineTransformer(n_knots=8, degree=3, knots="uniform", extrapolation="constant"))
    model = make_pipeline(ColumnTransformer([("s", splines, FEATURES)], remainder="passthrough"),
                          RidgeCV(alphas=np.logspace(-2, 3, 12)))
    model.fit(X, y, ridgecv__sample_weight=w)
    return [model.predict(o) for o in outs]


def _early_stopping_split(y, w, seed):
    own = np.flatnonzero(w == 1.0)
    fit, stop = train_test_split(own, test_size=0.1, stratify=y[own], random_state=seed)
    return np.concatenate([fit, np.flatnonzero(w != 1.0)]), stop


def lgbm(X, y, w, outs, cfg, seed, params):
    import lightgbm as lgb
    fit, stop = _early_stopping_split(y, w, seed)
    model = lgb.LGBMRegressor(n_estimators=cfg["trees"], learning_rate=0.02, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, reg_lambda=5.0, random_state=seed, n_jobs=4, verbose=-1,
                              monotone_constraints=[int(c == "alcohol") for c in X.columns], **params)
    model.fit(X.iloc[fit], y[fit], sample_weight=w[fit], eval_set=[(X.iloc[stop], y[stop])],
              callbacks=[lgb.early_stopping(200, verbose=False)])
    return [model.predict(o) for o in outs]


def catboost(X, y, w, outs, cfg, seed, classifier):
    from catboost import CatBoostClassifier, CatBoostRegressor
    fit, stop = _early_stopping_split(y, w, seed)
    common = {"iterations": cfg["trees"], "depth": 4, "l2_leaf_reg": 10.0, "random_seed": seed, "verbose": 0,
              "thread_count": 4, "allow_writing_files": False, "early_stopping_rounds": 200}
    model = (CatBoostClassifier(loss_function="MultiClass", learning_rate=0.05, **common) if classifier
             else CatBoostRegressor(learning_rate=0.03, **common))
    model.fit(X.iloc[fit], y[fit], sample_weight=w[fit], eval_set=(X.iloc[stop], y[stop]), use_best_model=True)
    if classifier:
        return [model.predict_proba(o) @ model.classes_.astype(float) for o in outs]
    return [model.predict(o) for o in outs]


def fit_arm(arm, X, y, ext, outs, cfg, seed):
    w = np.ones(len(y))
    if arm.endswith("_ext"):
        X = pd.concat([X.assign(ext=0.0), ext[FEATURES].assign(ext=1.0)], ignore_index=True)
        y = np.concatenate([y, ext[TARGET].to_numpy()])
        w = np.concatenate([w, np.full(len(ext), EXT_WEIGHT)])
        outs = [o.assign(ext=0.0) for o in outs]
        arm = arm.removesuffix("_ext")
    else:
        X = X.reset_index(drop=True)
    if arm in LGBM:
        return lgbm(X, y, w, outs, cfg, seed, LGBM[arm])
    if arm.startswith("catboost"):
        return catboost(X, y, w, outs, cfg, seed, classifier=arm.endswith("cls"))
    return {"alcohol_step": alcohol_step, "gam": gam}[arm](X, y, w, outs, cfg, seed)



def crossfit(y, u, folds, rng):
    pred = np.empty(len(y), dtype=int)
    per = []
    for k in np.unique(folds):
        m = folds == k
        pred[m] = to_labels(u[m], fit_cuts(y[~m], u[~m], rng))
        per.append(qwk(y[m], pred[m]))
    return qwk(y, pred), per, pred


def paired(a, b):
    d = np.asarray(a) - np.asarray(b)
    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
    return float(d.mean()), float(se)


def run(cfg, train, test, ext, sample=None):
    t_run = time.time()
    if cfg["rows"]:
        train = train.sample(cfg["rows"], random_state=SEED).reset_index(drop=True)
    X, y, Xt = train[FEATURES], train[TARGET].to_numpy(), test[FEATURES]
    R, K, n = cfg["repeats"], cfg["folds"], len(y)
    arms = [a for a in ARMS if ext is not None or not a.endswith("_ext")]
    oof = {a: np.zeros((R, n)) for a in arms}
    test_raw = {a: np.zeros(len(Xt)) for a in arms}
    seconds = {a: 0.0 for a in arms}
    folds = np.zeros((R, n), dtype=int)
    failed = {}
    for r in range(R):
        for k, (tr, va) in enumerate(StratifiedKFold(K, shuffle=True, random_state=SEED + r).split(X, y)):
            folds[r, va] = k
            for a in [a for a in arms if a not in failed]:
                t0 = time.time()
                try:
                    p_va, p_te = fit_arm(a, X.iloc[tr], y[tr], ext, [X.iloc[va], Xt], cfg, SEED + 100 * r + k)
                    assert np.isfinite(p_va).all() and np.isfinite(p_te).all(), "non-finite scores"
                except Exception as e:
                    failed[a] = f"{type(e).__name__}: {e}"
                    print(f"[{cfg['name']}] {a} dropped: {failed[a]}")
                    continue
                oof[a][r, va], seconds[a] = p_va, seconds[a] + time.time() - t0
                test_raw[a] += p_te / (R * K)
        print(f"[{cfg['name']}] repeat {r + 1}/{R} done, {time.time() - t_run:.0f}s")
    arms = [a for a in arms if a not in failed]
    members = [a for a in arms if a != REFERENCE and not a.endswith("_ext")]
    assert members, "every model arm failed"
    U = {a: np.vstack([percentile(oof[a][r]) for r in range(R)]) for a in arms}
    U["blend"] = np.vstack([percentile(np.mean([U[a][r] for a in members], 0)) for r in range(R)])
    Ut = {a: percentile(test_raw[a]) for a in arms}
    Ut["blend"] = percentile(np.mean([Ut[a] for a in members], 0))

    rng = np.random.default_rng(SEED)
    res = {}
    for a in U:
        out = [crossfit(y, U[a][r], folds[r], rng) for r in range(R)]
        res[a] = {"qwk": float(np.mean([o[0] for o in out])), "per_fold": [q for o in out for q in o[1]],
                  "labels": out[0][2]}
    ref = res[REFERENCE]["per_fold"]
    print(f"\n[{cfg['name']}] cross-fitted QWK ({R} x {K} folds; diff vs {REFERENCE}, paired over folds)")
    for a, v in sorted(res.items(), key=lambda kv: -kv[1]["qwk"]):
        d, se = paired(v["per_fold"], ref)
        print(f"  {a:18s} {v['qwk']:.5f}  fold sd {np.std(v['per_fold'], ddof=1):.4f}  diff {d:+.5f} +- {se:.5f}"
              f"  {seconds.get(a, 0.0):6.1f}s")
    for a in [a for a in arms if a.endswith("_ext")]:
        d, se = paired(res[a]["per_fold"], res[a.removesuffix("_ext")]["per_fold"])
        print(f"  external data, {a} vs {a.removesuffix('_ext')}: {d:+.5f} +- {se:.5f} (t {d / se if se > 0 else 0:+.2f})")

    singles = [a for a in arms if not a.endswith("_ext")]
    best = max(singles, key=lambda a: np.mean(res[a]["per_fold"]))
    d, se = paired(res["blend"]["per_fold"], res[best]["per_fold"])
    chosen = "blend" if d >= GATE_SE * se else best
    print(f"\n[{cfg['name']}] blend of {members} vs best single {best}: {d:+.5f} +- {se:.5f} -> submit {chosen}")

    cuts = fit_cuts(np.tile(y, R), U[chosen].ravel(), rng)
    labels = to_labels(Ut[chosen], cuts)
    share = lambda v: {int(c): round(float(np.mean(v == c)), 4) for c in LABELS}
    print(f"cuts (percentiles) {np.round(cuts, 4).tolist()}")
    print(f"label shares  train {share(y)}\n  OOF cross-fitted {share(res[chosen]['labels'])}\n  test {share(labels)}")
    lab = res[chosen]["labels"]
    for m in [m for m in PUBLIC_ROWS if m < n]:
        draws = [qwk(y[i], lab[i]) for i in (rng.choice(n, m, replace=False) for _ in range(300))]
        print(f"LB noise if the public part has {m} rows: QWK sd {np.std(draws):.4f}")
    if sample is None:
        print(f"[{cfg['name']}] done in {time.time() - t_run:.0f}s, nothing written")
        return

    sub = pd.DataFrame({ID: test[ID].to_numpy(), TARGET: labels.astype(int)})
    assert len(sub) == len(sample) and list(sub.columns) == list(sample.columns)
    assert (sub[ID].to_numpy() == sample[ID].to_numpy()).all() and sub[TARGET].isin(LABELS).all()
    extra = OUT / "extra"
    extra.mkdir(exist_ok=True)
    o = pd.DataFrame({ID: train[ID], TARGET: y, **{f"fold_r{r}": folds[r] for r in range(R)},
                      **{f"{a}_r{r}": U[a][r] for a in U for r in range(R)}})
    o.to_csv(extra / f"oof_{VERSION}.csv", index=False)
    pd.DataFrame({ID: test[ID], **{a: test_raw[a] for a in arms}, **{f"{a}_pct": Ut[a] for a in Ut}}).to_csv(
        extra / f"test_scores_{VERSION}.csv", index=False)
    summary = {"version": VERSION, "chosen": chosen, "best_single": best, "blend_members": members,
               "blend_vs_best": [d, se], "cuts": cuts.tolist(), "failed": failed,
               "arms": {a: {"qwk": v["qwk"], "per_fold": v["per_fold"]} for a, v in res.items()}}
    (extra / f"summary_{VERSION}.json").write_text(json.dumps(summary, indent=1))
    sub.to_csv(OUT / "submission.csv", index=False)
    print(f"[{cfg['name']}] done in {time.time() - t_run:.0f}s\nSUBMIT: {OUT / 'submission.csv'} ({chosen}, "
          f"cross-fitted QWK {res[chosen]['qwk']:.5f})")


def main():
    train, test, sample, ext = load()
    run(SMOKE, train, test, ext)
    run(FULL, train, test, ext, sample)


if __name__ == "__main__":
    main()
