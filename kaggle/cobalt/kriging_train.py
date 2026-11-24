import json
import time
from functools import partial

import numpy as np
import pandas as pd
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingRegressor

from kriging_preprocess import OUT, describe, load, log

N_REPEATS = 50
SEED = 0
KERNELS = {0.5: lambda h: np.exp(-h), 1.5: lambda h: (1 + np.sqrt(3) * h) * np.exp(-np.sqrt(3) * h)}
GPS = {"ok": (0.5, ()), "rock": (0.5, ("Rock",)), "rock+land": (0.5, ("Rock", "Landuse")),
       "rock nu1.5": (1.5, ("Rock",))}
BLEND = "mean of gp"


def sigmoid(v):
    return 1.0 / (1.0 + np.exp(-v))


def unpack(p, n_re):
    return {"sill": np.exp(p[0]), "range": np.exp(p[1]), "ratio": sigmoid(p[2]), "angle": np.pi * sigmoid(p[3]),
            "nugget": np.exp(p[4]), "re": np.exp(p[5:5 + n_re])}


def cov(xa, fa, xb, fb, q, nu):
    c, s = np.cos(q["angle"]), np.sin(q["angle"])
    rot = np.array([[c, s], [-s, c]])
    scale = np.array([q["range"], q["range"] * q["ratio"]])
    h = np.sqrt(((((xa @ rot.T)[:, None] - (xb @ rot.T)[None]) / scale) ** 2).sum(-1))
    k = q["sill"] * KERNELS[nu](h)
    for j, tau in enumerate(q["re"]):
        k = k + tau * (fa[:, j][:, None] == fb[:, j][None])
    return k


def site_arrays(df, factors):
    return df[["Xloc", "Yloc"]].to_numpy(float), df[list(factors)].to_numpy().reshape(len(df), len(factors))


def solve(x, f, z, q, nu):
    L = cho_factor(cov(x, f, x, f, q, nu) + (q["nugget"] + 1e-8) * np.eye(len(z)), lower=True)
    ki1 = cho_solve(L, np.ones(len(z)))
    s = ki1.sum()
    beta = (ki1 @ z) / s
    alpha = cho_solve(L, z - beta)
    return L, s, beta, alpha


def neg_reml(p, x, f, z, nu):
    try:
        L, s, beta, alpha = solve(x, f, z, unpack(p, f.shape[1]), nu)
    except np.linalg.LinAlgError:
        return 1e10
    return np.log(np.diag(L[0])).sum() + 0.5 * np.log(s) + 0.5 * (z - beta) @ alpha


def fit_gp(df, y, nu, factors):
    mu, sd = y.mean(), y.std()
    z = (y - mu) / sd
    x, f = site_arrays(df, factors)
    bounds = [(-9, 3), (np.log(0.03), np.log(20)), (-5, 5), (-5, 5), (-9, 3)] + [(-9, 3)] * len(factors)
    best = None
    for start_range in (0.3, 1.5):
        p0 = np.r_[np.log(0.6), np.log(start_range), 0.0, 0.0, np.log(0.3), np.full(len(factors), np.log(0.3))]
        r = minimize(neg_reml, p0, args=(x, f, z, nu), method="L-BFGS-B", bounds=bounds)
        if best is None or r.fun < best.fun:
            best = r
    q = unpack(best.x, len(factors))
    L, s, beta, alpha = solve(x, f, z, q, nu)
    return {"q": q, "nu": nu, "factors": factors, "x": x, "f": f, "L": L, "s": s, "beta": beta,
            "alpha": alpha, "mu": mu, "sd": sd, "neg_reml": best.fun}


def predict_gp(m, df):
    x, f = site_arrays(df, m["factors"])
    q = m["q"]
    k = cov(x, f, m["x"], m["f"], q, m["nu"])
    kik = cho_solve(m["L"], k.T)
    mean = m["beta"] + k @ m["alpha"]
    var = q["sill"] + q["re"].sum() + q["nugget"] - (k * kik.T).sum(1) + (1 - kik.sum(0)) ** 2 / m["s"]
    return m["mu"] + m["sd"] * mean, m["sd"] ** 2 * np.maximum(var, 0.0)


def gp(kind, scale, tr, te):
    y = tr["Co"].to_numpy(float)
    m = fit_gp(tr, np.log(y) if scale == "log" else y, *GPS[kind])
    mean, var = predict_gp(m, te)
    return (np.exp(mean + var / 2) if scale == "log" else mean), m


def rock_mean(tr, te):
    return te["Rock"].map(tr.groupby("Rock")["Co"].mean()).fillna(tr["Co"].mean()).to_numpy(), None


def gbdt_features(d):
    return np.column_stack([d["Xloc"], d["Yloc"], d["Xloc"] + d["Yloc"], d["Xloc"] - d["Yloc"], d["Landuse"], d["Rock"]])


def gbdt(tr, te):
    m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=8, min_samples_leaf=10,
                                      categorical_features=[4, 5], random_state=0)
    return m.fit(gbdt_features(tr), tr["Co"]).predict(gbdt_features(te)), None


CANDIDATES = {"rock mean": rock_mean, "gbdt": gbdt}
CANDIDATES.update({f"gp {kind} {scale}": partial(gp, kind, scale) for kind in GPS for scale in ("raw", "log")})


def predict_all(tr, te):
    out = {n: f(tr, te) for n, f in CANDIDATES.items()}
    preds = {n: p for n, (p, _) in out.items()}
    preds[BLEND] = np.mean([p for n, p in preds.items() if n.startswith("gp ")], 0)
    return preds, {n: m for n, (_, m) in out.items() if m is not None}


def one_repeat(r, train, n_hold):
    hold = np.random.default_rng(SEED + r).permutation(len(train))[:n_hold]
    te, tr = train.iloc[hold], train.drop(train.index[hold])
    preds, _ = predict_all(tr, te)
    y = te["Co"].to_numpy()
    return {n: float(np.mean((p - y) ** 2)) for n, p in preds.items()}


def params(m):
    q = {k: (v.round(4).tolist() if isinstance(v, np.ndarray) else round(float(v), 4)) for k, v in m["q"].items()}
    return {**q, "factors": list(m["factors"]), "nu": m["nu"], "y_sd": round(float(m["sd"]), 4),
            "neg_reml": round(float(m["neg_reml"]), 3)}


if __name__ == "__main__":
    train, test = load()
    describe(train, test)
    n_hold = int(round(len(train) * len(test) / (len(train) + len(test))))
    log(f"CV: {N_REPEATS} repeats, hold out {n_hold} of {len(train)} sites each (test density {len(test)}/"
        f"{len(train) + len(test)})")
    rows = []
    for r in range(N_REPEATS):
        t = time.time()
        rows.append(one_repeat(r, train, n_hold))
        if r == 0:
            log(f"repeat 0 took {time.time() - t:.1f} s: " + ", ".join(f"{n} {v:.3f}" for n, v in rows[0].items()))
        elif (r + 1) % 10 == 0:
            log(f"{r + 1}/{N_REPEATS} repeats; running mean MSE: "
                + ", ".join(f"{n} {v:.3f}" for n, v in pd.DataFrame(rows).mean().sort_values().items()))
    cv = pd.DataFrame(rows)
    best = cv.mean().idxmin()
    diff = cv.sub(cv[best], axis=0)
    table = pd.DataFrame({"cv_mse": cv.mean(), "se": cv.std() / np.sqrt(len(cv)),
                          "minus_best": diff.mean(), "paired_se": diff.std() / np.sqrt(len(cv))})
    log(f"CV MSE (mean over repeats; minus_best is paired against {best}):\n"
        f"{table.sort_values('cv_mse').round(4).to_string()}")

    preds, models = predict_all(train, test)
    for n, m in models.items():
        log(f"{n} full-data params: {params(m)}")
    pred = preds[best]
    log(f"chosen: {best}, CV MSE {table.loc[best, 'cv_mse']:.4f}")
    log(f"test predictions by Rock, mean of chosen: {pd.Series(pred).groupby(test['Rock'].to_numpy()).mean().round(2).to_dict()}; "
        f"correlation with other candidates: {pd.DataFrame(preds).corr()[best].round(3).to_dict()}")

    sub = pd.DataFrame({"V1": test["id"].to_numpy(), "V2": pred})
    assert len(sub) == 100 and (sub["V1"] == np.arange(1, 101)).all()
    assert np.isfinite(sub["V2"]).all() and (sub["V2"] > 0).all() and sub["V2"].max() < 3 * train["Co"].max()
    sub.to_csv(OUT / "submission.csv", index=False)
    (OUT / "extra").mkdir(exist_ok=True)
    cv.to_csv(OUT / "extra" / "cv_mse.csv", index=False)
    pd.DataFrame({"V1": test["id"], **preds}).to_csv(OUT / "extra" / "test_preds.csv", index=False)
    (OUT / "extra" / "choice.json").write_text(json.dumps(
        {"chosen": best, "cv": table.round(5).to_dict("index"), "params": {n: params(m) for n, m in models.items()}},
        indent=1))
    log(f"SUBMIT: {OUT / 'submission.csv'}")
