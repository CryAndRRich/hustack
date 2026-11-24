import math

import numpy as np
from scipy.linalg import cho_solve, solve_triangular
from scipy.special import logsumexp, ndtr

THRESH = 1.0
ASH_GRID = np.array([0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 13.0, 20.0])


def tail(m, s):
    return ndtr((-THRESH - m) / s) + ndtr((m - THRESH) / s)


def ols(S):
    A, b = S["A"], S["b"]
    Ainv = np.linalg.inv(A)
    bh = Ainv @ b
    dfr = S["dof"] - len(b)
    s2 = (S["yy"] - bh @ b) / dfr
    se = np.sqrt(s2 * np.diag(Ainv))
    return {"bh": bh, "se": se, "sigma2": s2, "dfr": dfr, "prob": tail(bh, se)}


def eb_ash(S, n_em=1000):
    o = ols(S)
    bh, se = o["bh"], o["se"]
    var = se[:, None] ** 2 + ASH_GRID[None, :] ** 2
    loglik = -0.5 * (np.log(2 * np.pi * var) + bh[:, None] ** 2 / var)
    pi = np.full(len(ASH_GRID), 1.0 / len(ASH_GRID))
    for _ in range(n_em):
        lr = np.log(pi + 1e-300)[None, :] + loglik
        R = np.exp(lr - logsumexp(lr, axis=1, keepdims=True))
        pi = R.mean(0)
    lr = np.log(pi + 1e-300)[None, :] + loglik
    R = np.exp(lr - logsumexp(lr, axis=1, keepdims=True))
    g2 = ASH_GRID[None, 1:] ** 2
    V = 1.0 / (1.0 / se[:, None] ** 2 + 1.0 / g2)
    M = V * bh[:, None] / se[:, None] ** 2
    prob = (R[:, 1:] * tail(M, np.sqrt(V))).sum(1)
    return {"prob": prob, "pi": pi, "post_mean": (R[:, 1:] * M).sum(1), "sigma2": o["sigma2"]}


def ash_draw(pi, n, rng):
    k = rng.choice(len(ASH_GRID), size=n, p=pi / pi.sum())
    return ASH_GRID[k] * rng.standard_normal(n)


def _logistic(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def ss_gibbs(S, n_iter, burn, seed, thin=20, a0=1.0, b0=1.0):
    A, b, yy, dof = S["A"], S["b"], S["yy"], S["dof"]
    p = len(b)
    d = np.diag(A).tolist()
    rng = np.random.default_rng(seed)
    beta = np.zeros(p)
    v = b.copy()
    sigma2, w, tau2 = yy / dof, 0.05, 4.0
    M, V, P1 = np.empty(p), np.empty(p), np.empty(p)
    acc_prob, acc_pip, acc_beta = np.zeros(p), np.zeros(p), np.zeros(p)
    kept, draws, trace = 0, [], []
    for it in range(n_iter):
        u = rng.random(p)
        e = rng.standard_normal(p)
        logodds = math.log(w) - math.log1p(-w)
        for j in rng.permutation(p).tolist():
            bj = beta[j]
            z = v[j] + d[j] * bj
            Vj = 1.0 / (d[j] / sigma2 + 1.0 / tau2)
            Mj = Vj * z / sigma2
            pj = _logistic(logodds + 0.5 * math.log(Vj / tau2) + 0.5 * Mj * Mj / Vj)
            M[j], V[j], P1[j] = Mj, Vj, pj
            new = Mj + math.sqrt(Vj) * e[j] if u[j] < pj else 0.0
            if new != bj:
                v -= A[j] * (new - bj)
                beta[j] = new
        on = beta != 0
        k = int(on.sum())
        rss = max(yy - beta @ b - beta @ v, 1e-9)
        sigma2 = 0.5 * rss / rng.gamma(0.5 * dof)
        w = rng.beta(1 + k, 1 + p - k)
        tau2 = (b0 + 0.5 * float(beta[on] @ beta[on])) / rng.gamma(a0 + 0.5 * k)
        if it >= burn:
            acc_prob += P1 * tail(M, np.sqrt(V))
            acc_pip += P1
            acc_beta += beta
            kept += 1
            trace.append((sigma2, w, tau2, k))
            if (it - burn) % thin == 0:
                draws.append((beta.copy(), sigma2))
    tr = np.array(trace)
    return {"prob": acc_prob / kept, "pip": acc_pip / kept, "post_mean": acc_beta / kept, "draws": draws,
            "sigma2": float(tr[:, 0].mean()), "w": float(tr[:, 1].mean()), "tau2": float(tr[:, 2].mean()),
            "k": float(tr[:, 3].mean())}


def hs_gibbs(S, n_iter, burn, seed, thin=20):
    A, b, yy, dof = S["A"], S["b"], S["yy"], S["dof"]
    p = len(b)
    rng = np.random.default_rng(seed)
    lam2, nu = np.ones(p), np.ones(p)
    tau2, xi, sigma2 = 1.0, 1.0, yy / dof
    acc_prob, acc_beta = np.zeros(p), np.zeros(p)
    kept, draws, trace = 0, [], []
    for it in range(n_iter):
        Ast = A + np.diag(1.0 / (lam2 * tau2))
        L = np.linalg.cholesky(Ast)
        mean = cho_solve((L, True), b)
        beta = mean + math.sqrt(sigma2) * solve_triangular(L.T, rng.standard_normal(p), lower=False)
        rss = max(yy - 2 * beta @ b + beta @ A @ beta, 1e-9)
        q = beta ** 2 / (lam2 * tau2)
        sigma2 = 0.5 * (rss + q.sum()) / rng.gamma(0.5 * (dof + p))
        lam2 = np.clip((1.0 / nu + beta ** 2 / (2 * tau2 * sigma2)) / rng.gamma(1.0, size=p), 1e-10, 1e10)
        tau2 = float(np.clip((1.0 / xi + (beta ** 2 / lam2).sum() / (2 * sigma2)) / rng.gamma(0.5 * (p + 1)), 1e-10, 1e10))
        nu = (1.0 + 1.0 / lam2) / rng.gamma(1.0, size=p)
        xi = (1.0 + 1.0 / tau2) / rng.gamma(1.0)
        if it >= burn:
            acc_prob += np.abs(beta) >= THRESH
            acc_beta += beta
            kept += 1
            trace.append((sigma2, tau2))
            if (it - burn) % thin == 0:
                draws.append((beta.copy(), sigma2))
    tr = np.array(trace)
    return {"prob": acc_prob / kept, "post_mean": acc_beta / kept, "draws": draws,
            "sigma2": float(tr[:, 0].mean()), "tau2": float(tr[:, 1].mean())}


def run_arms(S, cfg, seed):
    o = ols(S)
    eb = eb_ash(S)
    ss = [ss_gibbs(S, cfg["iter"], cfg["burn"], seed + c) for c in range(cfg["chains"])]
    hs = [hs_gibbs(S, cfg["iter"], cfg["burn"], seed + 100 + c) for c in range(cfg["chains"])]
    probs = {"ols": o["prob"], "eb": eb["prob"],
             "ss": np.mean([r["prob"] for r in ss], 0), "hs": np.mean([r["prob"] for r in hs], 0)}
    probs["blend"] = (probs["ss"] + probs["hs"] + probs["eb"]) / 3
    return probs, {"ols": o, "eb": eb, "ss": ss, "hs": hs}
