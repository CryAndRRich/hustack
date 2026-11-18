import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize


def sigmoid(v):
    return 1.0 / (1.0 + np.exp(-v))


def unpack(p, n_re):
    return {"sill": np.exp(p[0]), "range": np.exp(p[1]), "ratio": sigmoid(p[2]), "angle": np.pi * sigmoid(p[3]),
            "nugget": np.exp(p[4]), "re": np.exp(p[5:5 + n_re])}


def cov(xa, fa, xb, fb, q):
    c, s = np.cos(q["angle"]), np.sin(q["angle"])
    rot = np.array([[c, s], [-s, c]])
    scale = np.array([q["range"], q["range"] * q["ratio"]])
    h = np.sqrt(((((xa @ rot.T)[:, None] - (xb @ rot.T)[None]) / scale) ** 2).sum(-1))
    k = q["sill"] * np.exp(-h)
    for j, tau in enumerate(q["re"]):
        k = k + tau * (fa[:, j][:, None] == fb[:, j][None])
    return k


def site_arrays(df, factors):
    return df[["Xloc", "Yloc"]].to_numpy(float), df[list(factors)].to_numpy().reshape(len(df), len(factors))


def solve(x, f, z, q):
    L = cho_factor(cov(x, f, x, f, q) + (q["nugget"] + 1e-8) * np.eye(len(z)), lower=True)
    ki1 = cho_solve(L, np.ones(len(z)))
    s = ki1.sum()
    beta = (ki1 @ z) / s
    alpha = cho_solve(L, z - beta)
    return L, s, beta, alpha


def neg_reml(p, x, f, z):
    try:
        L, s, beta, alpha = solve(x, f, z, unpack(p, f.shape[1]))
    except np.linalg.LinAlgError:
        return 1e10
    return np.log(np.diag(L[0])).sum() + 0.5 * np.log(s) + 0.5 * (z - beta) @ alpha


def fit_gp(df, y, factors):
    mu, sd = y.mean(), y.std()
    z = (y - mu) / sd
    x, f = site_arrays(df, factors)
    bounds = [(-9, 3), (np.log(0.03), np.log(20)), (-5, 5), (-5, 5), (-9, 3)] + [(-9, 3)] * len(factors)
    best = None
    for start_range in (0.3, 1.5):
        p0 = np.r_[np.log(0.6), np.log(start_range), 0.0, 0.0, np.log(0.3), np.full(len(factors), np.log(0.3))]
        r = minimize(neg_reml, p0, args=(x, f, z), method="L-BFGS-B", bounds=bounds)
        if best is None or r.fun < best.fun:
            best = r
    q = unpack(best.x, len(factors))
    L, s, beta, alpha = solve(x, f, z, q)
    return {"q": q, "factors": factors, "x": x, "f": f, "L": L, "beta": beta, "alpha": alpha, "z": z, "mu": mu,
            "sd": sd, "neg_reml": best.fun}


def predict_gp(m, df):
    x, f = site_arrays(df, m["factors"])
    k = cov(x, f, m["x"], m["f"], m["q"])
    q = m["q"]
    kik = cho_solve(m["L"], k.T)
    var = q["sill"] + q["re"].sum() + q["nugget"] - (k * kik.T).sum(1)
    return m["mu"] + m["sd"] * (m["beta"] + k @ m["alpha"]), m["sd"] * np.sqrt(np.maximum(var, 0.0))


def block_loo(m, groups):
    kinv = cho_solve(m["L"], np.eye(len(m["z"])))
    mean, sd = np.empty(len(groups)), np.empty(len(groups))
    for g in np.unique(groups):
        b = np.flatnonzero(groups == g)
        cov_b = np.linalg.inv(kinv[np.ix_(b, b)])
        mean[b] = m["z"][b] - cov_b @ m["alpha"][b]
        sd[b] = np.sqrt(np.maximum(np.diag(cov_b), 0.0))
    return m["mu"] + m["sd"] * mean, m["sd"] * sd
