import io
import json
import zipfile

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from preprocess import CATS, NUM, load, out_dir

CFG = {
    "folds": 5, "seed": 0, "rows": None, "margin": 0.002,
    "omega": (0.50, 1.501, 0.01), "tau": (5.50, 7.501, 0.01),
    "start": {"omega": 1.0, "tau": 6.45, "x08": "log"},
    "ext": {"x01": 2.5, "x05": 1.5},
    "lgb": {"objective": "regression", "learning_rate": 0.03, "num_leaves": 31, "min_data_in_leaf": 40,
            "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0},
    "rounds": 5000, "stop": 200,
}
SMOKE = {"rows": 3000, "folds": 2, "rounds": 50, "stop": 10}
X08 = {"log": np.log, "log1p": np.log1p, "sqrt": np.sqrt, "lin": lambda v: v}
KINDS = {"formula": "base", "formula_wide": "wide", "formula_cat": "cat"}
ARMS = (*KINDS, "exact", "formula_gbm", "lgb")
X01_MAX = 3.0
EXACT = {"p": {"omega": 1.0, "tau": 6.5, "x08": "log"},
         "coef": {"cat_beta": 1.5, "cat_gamma": -1.0, "cat_delta": 0.5, "cat_epsilon": 2.5, "x07_gamma": 2.0,
                  "x07_delta": -1.5, "x01_sin": 3.0, "x05_sq": 1.2, "x04_step": 2.5, "x06": 0.8, "x08": 1.5,
                  "x02_x03": 2.0}}


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def terms(df, p, kind):
    x = {c: df[c].to_numpy(float) for c in NUM}
    base = {"x01_sin": np.sin(p["omega"] * x["x01"]), "x05_sq": x["x05"] ** 2,
            "x04_step": np.where(np.isnan(x["x04"]), np.nan, x["x04"] > p["tau"]),
            "x06": x["x06"], "x08": X08[p["x08"]](x["x08"]), "x02_x03": x["x02"] * x["x03"]}
    if kind == "wide":
        base.update({"x01_cos": np.cos(p["omega"] * x["x01"]), "x01": x["x01"], "x05": x["x05"], "x04": x["x04"]})
    groups = [g for g in (np.abs(x["x01"]) > X01_MAX, np.abs(x["x01"]) <= X01_MAX) if g.any()]
    for k, v in base.items():
        v = base[k] = v.astype(float)
        for g in groups:
            v[g & np.isnan(v)] = np.nanmean(v[g])
    A = (df.cat_a.to_numpy()[:, None] == np.array(CATS["cat_a"])[None]).astype(float)
    cols = {}
    for k, lv in enumerate(CATS["cat_a"]):
        cols[f"cat_{lv}"] = A[:, k]
        cols[f"x07_{lv}"] = A[:, k] * x["x07"]
        if kind == "cat":
            cols.update({f"{t}_{lv}": A[:, k] * v for t, v in base.items()})
    if kind != "cat":
        cols.update(base)
    return pd.DataFrame(cols, index=df.index)


def exact(df):
    X = terms(df, EXACT["p"], "base")
    assert set(EXACT["coef"]) <= set(X.columns)
    return X.to_numpy() @ np.array([EXACT["coef"].get(c, 0.0) for c in X.columns])


class Formula:
    def __init__(self, cfg, kind):
        self.grid = {"omega": np.arange(*cfg["omega"]), "tau": np.arange(*cfg["tau"]), "x08": list(X08)}
        self.kind = kind
        self.p = dict(cfg["start"])

    def _solve(self, df, y, p):
        X = terms(df, p, self.kind)
        beta = np.linalg.lstsq(X.to_numpy(), y, rcond=None)[0]
        return float(np.sum((y - X.to_numpy() @ beta) ** 2)), beta, list(X.columns)

    def fit(self, df, y, passes=2):
        for _ in range(passes):
            for k, grid in self.grid.items():
                self.p[k] = min(grid, key=lambda v: self._solve(df, y, {**self.p, k: v})[0])
        self.p = {k: v if isinstance(v, str) else round(float(v), 4) for k, v in self.p.items()}
        rss, self.beta, self.cols = self._solve(df, y, self.p)
        self.sigma = float(np.sqrt(rss / (len(y) - len(self.beta))))
        return self

    def predict(self, df):
        return terms(df, self.p, self.kind).to_numpy() @ self.beta


def lgb_frame(df):
    X = df[NUM + list(CATS)].copy()
    for c, levels in CATS.items():
        X[c] = pd.Categorical(X[c], categories=levels)
    return X


def fit_lgb(cfg, X, y, Xva=None, yva=None, rounds=None):
    params = {**cfg["lgb"], "seed": cfg["seed"], "verbose": -1}
    if Xva is None:
        return lgb.train(params, lgb.Dataset(X, y), rounds)
    return lgb.train(params, lgb.Dataset(X, y), cfg["rounds"], valid_sets=[lgb.Dataset(Xva, yva)],
                     callbacks=[lgb.early_stopping(cfg["stop"], verbose=False)])


def run_split(cfg, df, a, b):
    y, ta, tb = df.target.to_numpy(), df.iloc[a], df.iloc[b]
    fits = {n: Formula(cfg, k).fit(ta, y[a]) for n, k in KINDS.items()}
    out, f = {n: m.predict(tb) for n, m in fits.items()}, fits["formula"]
    out["exact"] = exact(tb)
    iters = {}
    m = fit_lgb(cfg, lgb_frame(ta), y[a] - f.predict(ta), lgb_frame(tb), y[b] - out["formula"])
    out["formula_gbm"], iters["formula_gbm"] = out["formula"] + m.predict(lgb_frame(tb), num_iteration=m.best_iteration), m.best_iteration
    m = fit_lgb(cfg, lgb_frame(ta), y[a], lgb_frame(tb), y[b])
    out["lgb"], iters["lgb"] = m.predict(lgb_frame(tb), num_iteration=m.best_iteration), m.best_iteration
    return out, iters


def diagnose(df, r):
    r = pd.Series(r, index=df.index)
    rows = []
    for c in NUM + list(CATS) + ["cat_a×cat_b"]:
        key = df.cat_a + "/" + df.cat_b if c == "cat_a×cat_b" else df[c]
        if c in NUM and df[c].nunique() > 13:
            key = pd.qcut(df[c], 10, duplicates="drop")
        g = r.groupby(key, observed=True)
        t = g.mean() / (g.std() / np.sqrt(g.size()))
        miss = r[df[c].isna()].mean() if c in NUM and df[c].isna().any() else None
        rows.append({"feature": c, "max_t": round(float(t.abs().max()), 2), "bin_means": g.mean().round(3).tolist(),
                     "missing_mean": None if miss is None else round(float(miss), 3)})
    rows.sort(key=lambda d: -d["max_t"])
    print("residual diagnostics (formula oof), sorted by max |t| over bins:")
    for d in rows:
        print(f"  {d['feature']:12s} max|t| {d['max_t']:5.2f}  missing {d['missing_mean']}  {d['bin_means']}")
    return rows


def run(cfg, write=True):
    tr, te = load()
    if cfg["rows"]:
        tr = tr.sample(cfg["rows"], random_state=cfg["seed"]).reset_index(drop=True)
    y = tr.target.to_numpy()

    oof, iters = {n: np.zeros(len(tr)) for n in ARMS}, {"formula_gbm": [], "lgb": []}
    for a, b in KFold(cfg["folds"], shuffle=True, random_state=cfg["seed"]).split(tr):
        out, it = run_split(cfg, tr, a, b)
        for n in ARMS:
            oof[n][b] = out[n]
        for n in it:
            iters[n].append(it[n])
    kf = {n: rmse(y, oof[n]) for n in ARMS}

    e = cfg["ext"]
    inner = (tr.x01.abs() < e["x01"]) & ~(tr.x05 >= e["x05"])
    a, b = np.flatnonzero(inner), np.flatnonzero(~inner)
    ext, _ = run_split(cfg, tr, a, b)
    parts = {"all": np.ones(len(b), bool), "x01_out": tr.x01.abs().to_numpy()[b] >= e["x01"],
             "x05_out": tr.x05.to_numpy()[b] >= e["x05"]}
    ex = {n: {k: rmse(y[b][m], ext[n][m]) for k, m in parts.items()} for n in ARMS}
    print(f"ext fold: fit on {len(a)} rows inside |x01|<{e['x01']}, x05<{e['x05']}; scored on {len(b)} "
          f"({parts['x01_out'].sum()} x01 out, {parts['x05_out'].sum()} x05 out)")
    print(f"{'arm':12s} {'kfold':>8s} {'ext all':>8s} {'ext x01':>8s} {'ext x05':>8s}")
    for n in ARMS:
        print(f"{n:12s} {kf[n]:8.4f} " + " ".join(f"{ex[n][k]:8.4f}" for k in parts))
    print("lgb best iterations per fold:", iters)
    diag = diagnose(tr, y - oof["formula"])

    full = {n: Formula(cfg, k).fit(tr, y) for n, k in KINDS.items()}
    pred = {n: f.predict(te) for n, f in full.items()}
    pred["exact"] = exact(te)
    for n, f in full.items():
        print(f"{n}: params {f.p}, residual sd {f.sigma:.4f}")
    print("formula coefficients:", {c: round(float(v), 4) for c, v in zip(full["formula"].cols, full["formula"].beta)})
    rounds = {n: max(1, int(np.mean(v))) for n, v in iters.items()}
    m = fit_lgb(cfg, lgb_frame(tr), y - full["formula"].predict(tr), rounds=rounds["formula_gbm"])
    pred["formula_gbm"] = pred["formula"] + m.predict(lgb_frame(te))
    imp = pd.Series(m.feature_importance("gain"), index=m.feature_name()).sort_values(ascending=False)
    print("residual gbm gain share:", (imp / max(imp.sum(), 1e-12)).round(3).head(8).to_dict())
    pred["lgb"] = fit_lgb(cfg, lgb_frame(tr), y, rounds=rounds["lgb"]).predict(lgb_frame(te))

    ship = "exact"
    for n in KINDS:
        if kf[n] < kf[ship] - cfg["margin"] and ex[n]["all"] <= ex[ship]["all"]:
            ship = n
    for n in ARMS:
        assert np.isfinite(pred[n]).all() and len(pred[n]) == len(te), n
    print(f"ship {ship}: kfold {kf[ship]:.4f}, ext {ex[ship]['all']:.4f}; test pred mean {pred[ship].mean():.3f} sd {pred[ship].std():.3f}")
    if not write:
        return
    od = out_dir()
    pd.DataFrame({"id": te.id, "target": pred[ship]}).to_csv(od / "submission.csv", index=False)
    with zipfile.ZipFile(od / "extra_v1.1.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for name, frame in (("oof.csv", pd.DataFrame({"id": tr.id, "target": y, **oof})),
                            ("ext.csv", pd.DataFrame({"id": tr.id.to_numpy()[b], "target": y[b], **ext})),
                            ("test_pred.csv", pd.DataFrame({"id": te.id, **pred}))):
            buf = io.StringIO()
            frame.to_csv(buf, index=False)
            z.writestr(name, buf.getvalue())
        z.writestr("report.json", json.dumps({
            "kfold": kf, "ext": ex, "ship": ship, "iters": iters, "diagnostics": diag,
            "fits": {n: {"params": f.p, "sigma": f.sigma, "coef": dict(zip(f.cols, map(float, f.beta)))}
                     for n, f in full.items()}}, indent=1))
    print(f"SUBMIT: submission.csv ({ship}); extras in extra_v1.1.zip")


def main():
    run({**CFG, **SMOKE}, write=False)
    print("smoke ok")
    run(CFG)


if __name__ == "__main__":
    main()
