import time
from pathlib import Path

import numpy as np
import pandas as pd

from preprocess import DIAM, GAP4, TARGET, features, load

CFG = {"folds": 5, "seed": 0, "monitor": 0.2, "rounds": 3000, "patience": 200, "gate_t": 2.0,
       "lgb": {"objective": "regression", "learning_rate": 0.02, "num_leaves": 15, "min_data_in_leaf": 40,
               "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
               "verbose": -1, "num_threads": 4}}
OUT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")


def formula(G):
    return 200 + 2.8 * (G[:, 2] - G[:, 6]) + 2.2 * (G[:, 11] - G[:, 16])


def ols(A, y):
    A1 = np.column_stack([np.ones(len(A)), A])
    return np.linalg.lstsq(A1, y, rcond=None)[0]


def predict(coef, A):
    return coef[0] + A @ coef[1:]


def resid_inputs(df):
    S, G, sd = features(df)
    X = df[DIAM].to_numpy()
    return np.column_stack([S, G, X, sd, X.mean(1)]), G


def fit_resid(lgb, A, r, rng):
    mon = rng.random(len(r)) < CFG["monitor"]
    dfit, dmon = lgb.Dataset(A[~mon], r[~mon]), lgb.Dataset(A[mon], r[mon])
    m = lgb.train({**CFG["lgb"], "seed": int(rng.integers(1 << 30))}, dfit, CFG["rounds"], valid_sets=[dmon],
                  callbacks=[lgb.early_stopping(CFG["patience"], verbose=False)])
    return m


def validate(y, G, A):
    fold = np.random.default_rng(CFG["seed"]).permutation(len(y)) % CFG["folds"]
    f = formula(G)
    oof = {"gap4": np.zeros(len(y)), "formula": f}
    for k in range(CFG["folds"]):
        fit, ev = fold != k, fold == k
        oof["gap4"][ev] = predict(ols(G[fit][:, list(GAP4)], y[fit]), G[ev][:, list(GAP4)])
    iters = []
    try:
        import lightgbm as lgb
        t0 = time.time()
        rng = np.random.default_rng(CFG["seed"] + 1)
        oof["resid"] = np.zeros(len(y))
        for k in range(CFG["folds"]):
            fit, ev = fold != k, fold == k
            m = fit_resid(lgb, A[fit], (y - f)[fit], rng)
            iters.append(m.best_iteration)
            oof["resid"][ev] = f[ev] + m.predict(A[ev], num_iteration=m.best_iteration)
        print(f"lightgbm residual: best iterations per fold {iters}, {time.time() - t0:.1f}s")
    except Exception as e:
        print(f"lightgbm residual skipped: {e!r}")
        oof.pop("resid", None)
    rows = []
    for a, p in oof.items():
        e2 = (p - y) ** 2
        d = e2 - (f - y) ** 2
        per_fold = [np.sqrt(e2[fold == k].mean()) for k in range(CFG["folds"])]
        rows.append({"arm": a, "rmse": np.sqrt(e2.mean()), "fold_sd": np.std(per_fold),
                     "d_mse_vs_formula": d.mean(), "t": -d.mean() / (d.std(ddof=1) / np.sqrt(len(d)) + 1e-300),
                     "folds_better": int(sum(d[fold == k].mean() < 0 for k in range(CFG["folds"])))})
    res = pd.DataFrame(rows)
    print(f"{CFG['folds']}-fold out-of-fold (formula fits nothing; t > 0 means better than formula):")
    print(res.to_string(index=False, float_format=lambda v: f"{v:.6f}"))
    return res, oof, iters


def main():
    t0 = time.time()
    tr, te = load()
    y = tr[TARGET].to_numpy()
    A, G = resid_inputs(tr)
    At, Gt = resid_inputs(te)
    res, oof, iters = validate(y, G, A)

    test_pred = {"gap4": predict(ols(G[:, list(GAP4)], y), Gt[:, list(GAP4)]), "formula": formula(Gt)}
    if "resid" in oof:
        import lightgbm as lgb
        n_iter = max(1, int(np.median(iters)))
        m = lgb.train({**CFG["lgb"], "seed": CFG["seed"]}, lgb.Dataset(A, y - formula(G)), n_iter)
        test_pred["resid"] = test_pred["formula"] + m.predict(At)
        r = res.set_index("arm").loc["resid"]
        gate = r["t"] > CFG["gate_t"]
        print(f"V2 candidate (formula + lightgbm residual, {n_iter} rounds): t {r['t']:.2f}, "
              f"gate t > {CFG['gate_t']}: {'PASS' if gate else 'FAIL'}; "
              f"rms change vs formula on test {np.sqrt(np.mean((test_pred['resid'] - test_pred['formula']) ** 2)):.5f}")
    print(f"rms(test gap4 - formula) {np.sqrt(np.mean((test_pred['gap4'] - test_pred['formula']) ** 2)):.5f}")

    extra = OUT / "extra"
    extra.mkdir(exist_ok=True)
    res.to_csv(extra / "validation.csv", index=False)
    pd.DataFrame({"id": tr["id"], TARGET: y, **oof}).to_csv(extra / "oof.csv", index=False)
    pd.DataFrame({"id": te["id"], **test_pred}).to_csv(extra / "test_preds.csv", index=False)
    if "resid" in test_pred:
        pd.DataFrame({"id": te["id"], TARGET: test_pred["resid"]}).to_csv(extra / "submission_resid.csv", index=False)

    sub = pd.DataFrame({"id": te["id"], TARGET: test_pred["formula"]})
    assert len(sub) == 4000 and sub["id"].tolist() == te["id"].tolist()
    assert np.isfinite(sub[TARGET]).all()
    sub.to_csv(OUT / "submission.csv", index=False)
    rmse = float(res.loc[res["arm"] == "formula", "rmse"].iloc[0])
    print(f"submitted arm formula: train rmse {rmse:.5f} (no fit)")
    print(f"done in {time.time() - t0:.1f}s")
    print(f"SUBMIT: {OUT / 'submission.csv'}")


if __name__ == "__main__":
    main()
