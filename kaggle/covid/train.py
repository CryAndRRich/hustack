import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from deconv import LAMS, deconv_features
from preprocess import FEATS, TARGET, build_panel, load

W3 = tuple(range(1, 8)) + (14, 21)
CFG = {"sets": {"dc": ("base", "w3", "s7", "dc"), "dco": ("base", "l7", "dc"), "dck": ("base", "l7", "dc", "kf")},
       "alphas": (10.0, 30.0, 100.0, 300.0), "stack_alpha": 30.0, "hgb_set": "dck",
       "folds": 30, "ref_folds": 10, "ref": ("ridge_dco_a30", 0.86209)}
OUT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")


def features(panel):
    g = panel.groupby("seg")
    tp = g[TARGET]
    X = {"tp_lag1": tp.shift(1)}
    for j in W3:
        X[f"dtp_{j}"] = tp.shift(j) - tp.shift(j + 1)
    for f in FEATS:
        X[f] = panel[f]
        X[f"d_{f}"] = panel[f] - g[f].shift(1)
        X[f"d7_{f}"] = g[f].shift(7) - g[f].shift(8)
    X = pd.DataFrame(X)
    return X, panel[TARGET] - X["tp_lag1"]


def ridge(cols, a, seg=None):
    def fit_predict(X, y, Xt):
        ok = X[cols].notna().all(axis=1)
        w = None if seg is None else 1.0 / y[ok].groupby(seg).transform("var").to_numpy()
        assert w is None or np.isfinite(w).all()
        m = Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=a))])
        return m.fit(X.loc[ok, cols], y[ok], ridge__sample_weight=w).predict(Xt[cols])
    return fit_predict


def hgb(cols):
    def fit_predict(X, y, Xt):
        ok = X[cols].notna().all(axis=1)
        m = HistGradientBoostingRegressor(learning_rate=0.03, max_iter=300, max_leaf_nodes=8, min_samples_leaf=40,
                                          l2_regularization=1.0, early_stopping=False, random_state=0)
        return m.fit(X.loc[ok, cols], y[ok]).predict(Xt[cols])
    return fit_predict


def make_arms(X, dc_cols, kf_cols, seg):
    groups = {"base": ["tp_lag1"] + [c for f in FEATS for c in (f, f"d_{f}")],
              "w3": [f"dtp_{j}" for j in W3], "l7": [f"dtp_{j}" for j in range(1, 8)],
              "s7": [f"d7_{f}" for f in FEATS], "dc": list(dc_cols), "kf": list(kf_cols)}
    arms = {"persist": lambda X, y, Xt: np.zeros(len(Xt))}
    for lam in LAMS:
        arms[f"kf_{lam:g}"] = lambda X, y, Xt, lam=lam: (Xt[f"tr_{lam:g}"] - Xt[f"ep_{lam:g}"]).to_numpy()
    sets = {}
    for name, parts in CFG["sets"].items():
        sets[name] = [c for part in parts for c in groups[part]]
        assert all(c in X for c in sets[name]) and len(set(sets[name])) == len(sets[name])
        for a in CFG["alphas"]:
            arms[f"ridge_{name}_a{a:g}"] = ridge(sets[name], a)
            arms[f"wridge_{name}_a{a:g}"] = ridge(sets[name], a, seg)
    arms[f"hgb_{CFG['hgb_set']}"] = hgb(sets[CFG["hgb_set"]])
    members = [a for a in arms if a != "persist" and ("ridge_" not in a or a.endswith(f"_a{CFG['stack_alpha']:g}"))]
    return arms, members


def combine(P, members, stack):
    return P.assign(ens_mean=P[members].mean(axis=1), stack=stack)


def validate(X, y, end, arms, members):
    known = X.notna().all(axis=1) & y.notna()
    pred = {a: [] for a in arms}
    truth, fold = [], []
    for h in range(CFG["folds"]):
        fit, ev = y.notna() & (end > h), known & (end == h)
        truth.append(y[ev].to_numpy())
        fold.append(np.full(int(ev.sum()), h))
        for name, arm in arms.items():
            pred[name].append(arm(X[fit], y[fit], X[ev]))
    truth, fold = np.concatenate(truth), np.concatenate(fold)
    P = pd.DataFrame({a: np.concatenate(p) for a, p in pred.items()})
    stack = np.empty(len(P))
    for h in range(CFG["folds"]):
        w = nnls(P.loc[fold != h, members].to_numpy(), truth[fold != h])[0]
        stack[fold == h] = P.loc[fold == h, members].to_numpy() @ w
    P = combine(P, members, stack)
    w_all = nnls(P[members].to_numpy(), truth)[0]

    ref = CFG["ref"][0]
    rows = []
    for a in P:
        e, r = P[a].to_numpy() - truth, P[ref].to_numpy() - truth
        d = e ** 2 - r ** 2
        rows.append({"arm": a, "rmse": float(np.sqrt(np.mean(e ** 2))),
                     "rmse10": float(np.sqrt(np.mean(e[fold < CFG["ref_folds"]] ** 2))),
                     "d_mse_vs_ref": float(d.mean()), "se": float(d.std(ddof=1) / np.sqrt(len(d)))})
    res = pd.DataFrame(rows)
    print(f"validation: {len(truth)} one-day forecasts over the last {CFG['folds']} days per segment "
          f"(rmse10: the last {CFG['ref_folds']}, {int((fold < CFG['ref_folds']).sum())} forecasts); ref {ref}")
    print(res.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
    got = float(res.loc[res["arm"] == ref, "rmse10"].iloc[0])
    print(f"reference {ref}: rmse10 {got:.5f} (V2.1: {CFG['ref'][1]:.5f}) -> {'OK' if abs(got - CFG['ref'][1]) < 5e-5 else 'MISMATCH'}")
    print("stack weights (all folds): " + ", ".join(f"{m} {v:.3f}" for m, v in zip(members, w_all) if v > 0))
    return res, w_all


def main():
    t0 = time.time()
    df = load()
    panel = build_panel(df)
    X, y = features(panel)
    t1 = time.time()
    dc, kf = deconv_features(panel)
    print(f"deconvolution features {dc.shape[1]} + trend/noise features {kf.shape[1]} in {time.time() - t1:.1f}s")
    X = pd.concat([X, dc, kf], axis=1)
    end = panel.groupby("seg")["t"].transform("max") - panel["t"]
    arms, members = make_arms(X, dc.columns, kf.columns, panel["seg"])
    res, w_all = validate(X, y, end, arms, members)
    best = res.loc[res["rmse"].idxmin(), "arm"]

    known = y.notna()
    open_ = panel[TARGET].isna()
    assert X[open_].notna().to_numpy().all()
    P = pd.DataFrame({name: arm(X[known], y[known], X[open_]) for name, arm in arms.items()}, index=X.index[open_])
    P = combine(P, members, P[members].to_numpy() @ w_all)
    fc = (P + X.loc[open_, "tp_lag1"].to_numpy()[:, None]).assign(seg=panel.loc[open_, "seg"], t=panel.loc[open_, "t"])

    te = df[df["split"] == "test"].copy()
    te = te.merge(fc, on=["seg", "t"], how="left", validate="many_to_one")
    assert len(te) == 700 and (te["y_chain"].isna() == te[best].notna()).all()
    te["pred"] = te["y_chain"].fillna(te[best])
    sub = te[["id"]].assign(**{TARGET: te["pred"]})
    assert sub["id"].tolist() == df.loc[df["split"] == "test", "id"].tolist()
    assert np.isfinite(sub[TARGET]).all() and (sub[TARGET] > 0).all()

    extra = OUT / "extra"
    extra.mkdir(exist_ok=True)
    res.to_csv(extra / "validation.csv", index=False)
    te.loc[te["y_chain"].isna(), ["id", "seg", "t"] + list(P)].to_csv(extra / "chain_end_forecasts.csv", index=False)
    sub.to_csv(OUT / "submission.csv", index=False)
    n_open = int(te["y_chain"].isna().sum())
    r = res.set_index("arm").loc[best]
    print(f"chosen arm {best}: val rmse {r['rmse']:.4f} (last {CFG['ref_folds']} days {r['rmse10']:.4f}) on chain ends "
          f"-> expected LB rmse ≈ {r['rmse'] * np.sqrt(n_open / len(te)):.4f} ({n_open} forecast rows of {len(te)}, rest exact)")
    print(f"done in {time.time() - t0:.1f}s")
    print(f"SUBMIT: {OUT / 'submission.csv'}")


if __name__ == "__main__":
    main()
