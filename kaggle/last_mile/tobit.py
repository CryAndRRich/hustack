import numpy as np
from scipy.optimize import minimize
from scipy.special import log_ndtr

from runtime import log


def _nll(theta, D, y, cens, pen, lam):
    b, log_s = theta[:-1], theta[-1]
    s = np.exp(log_s)
    mu = D @ b
    z = (y - mu) / s
    u = ~cens
    c = z[cens]
    log_cdf = log_ndtr(c)
    m = np.exp(-0.5 * c * c - 0.5 * np.log(2 * np.pi) - log_cdf)
    n = len(y)
    nll = (0.5 * np.sum(z[u] ** 2) + u.sum() * log_s - np.sum(log_cdf)) / n + 0.5 * lam * np.sum((b * pen) ** 2)
    g_mu = np.zeros(n)
    g_mu[u] = -z[u] / s
    g_mu[cens] = m / s
    grad_b = D.T @ g_mu / n + lam * pen * pen * b
    grad_s = (np.sum(1 - z[u] ** 2) + np.sum(m * c)) / n
    return nll, np.r_[grad_b, grad_s]


def fit_tobit(D, y, floor, pen=None, lam=0.0):
    pen = np.zeros(D.shape[1]) if pen is None else pen
    cens = y <= floor + 1e-9
    y = np.where(cens, floor, y)
    b0 = np.linalg.lstsq(D, y, rcond=None)[0]
    s0 = np.std(y - D @ b0)
    res = minimize(_nll, np.r_[b0, np.log(s0)], args=(D, y, cens, pen, lam), jac=True, method="L-BFGS-B",
                   options={"maxiter": 5000, "gtol": 1e-9})
    if not res.success:
        log(f"    tobit optimiser: {res.message} (|grad| {np.abs(res.jac).max():.2e})")
    return res.x[:-1], float(np.exp(res.x[-1]))
