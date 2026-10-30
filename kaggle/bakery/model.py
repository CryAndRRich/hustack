import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize
from scipy.special import betainc, betaln, gammaln

FLOOR = 1e-250


def nb_loglik(eta, y, s, cens, r):
    mu = np.exp(eta)
    x = mu / (r + mu)
    lx, l1x = np.log(x), np.log1p(-x)
    ll = gammaln(y + r) - gammaln(r) - gammaln(y + 1) + r * l1x + y * lx
    g = r * (y - mu) / (r + mu)
    S = betainc(s[cens], r, x[cens])
    ok = S > FLOOR
    c = np.flatnonzero(cens)[ok]
    ll[c] = np.log(S[ok])
    g[c] = np.exp(s[c] * lx[c] + r * l1x[c] - betaln(s[c], r) - ll[c])
    return ll, g


def expected_sales(eta, s, r):
    mu = np.exp(eta)
    x = mu / (r + mu)
    out = np.zeros(len(eta))
    for k in range(1, int(s.max()) + 1):
        m = s >= k
        out[m] += betainc(k, r, x[m])
    return out


class CensoredNB:
    def __init__(self, sizes, lam, n_x, lam_x, n_a=0, lam_a=0.0):
        self.names = list(sizes)
        self.n = {**sizes, "x": n_x}
        self.off = dict(zip(self.names + ["x"], np.cumsum([0] + [self.n[b] for b in self.names])))
        self.size = sum(self.n.values())
        self.n_a = n_a
        self.lam = np.concatenate([np.full(sizes[b], lam[b]) for b in self.names] + [np.full(n_x, lam_x),
                                                                                    np.full(n_a, lam_a)])

    def design(self, idx, X):
        n = len(X)
        cols = np.stack([self.off[b] + idx[b] for b in self.names], 1)
        oh = sp.csr_matrix((np.ones(cols.size), (np.repeat(np.arange(n), len(self.names)), cols.ravel())),
                           shape=(n, self.off["x"]))
        return sp.hstack([oh, sp.csr_matrix(X)]).tocsr()

    def eta(self, Z, A, theta=None):
        th = self.theta if theta is None else theta
        mu = np.exp(Z @ th)
        if self.n_a:
            mu = mu + A @ self.gamma
        return np.log(mu)

    def fit(self, Z, A, y, s, warm, maxiter=3000):
        y, s = y.astype(float), s.astype(float)
        cens = y >= s
        k = self.size
        par = np.zeros(k + self.n_a + 1)
        par[k: k + self.n_a] = 0.5
        par[-1] = np.log(10.0)
        o, n = self.off[warm], self.n[warm]
        w = Z[:, o: o + n].tocoo()
        tot = np.bincount(w.col, weights=y[w.row], minlength=n)
        cnt = np.bincount(w.col, minlength=n)
        par[o: o + n] = np.log(np.maximum(tot, 1) / np.maximum(cnt, 1))
        ZT = Z.T.tocsr()

        def f(p):
            mb = np.exp(Z @ p[:k])
            mu = mb + (A @ p[k:-1] if self.n_a else 0.0)
            eta, r = np.log(mu), np.exp(p[-1])
            ll, g = nb_loglik(eta, y, s, cens, r)
            h = 1e-4
            dr = (nb_loglik(eta, y, s, cens, r * np.exp(h))[0].sum() - nb_loglik(eta, y, s, cens, r * np.exp(-h))[0].sum()) / (2 * h)
            gt = ZT @ (g * mb / mu)
            ga = A.T @ (g / mu) if self.n_a else np.zeros(0)
            pen = 0.5 * np.sum(self.lam * p[:-1] ** 2)
            grad = np.concatenate([-gt, -ga]) + self.lam * p[:-1]
            return -ll.sum() + pen, np.append(grad, -dr)

        bounds = [(None, None)] * k + [(0.0, None)] * self.n_a + [(None, None)]
        res = minimize(f, par, jac=True, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": maxiter, "maxfun": maxiter * 2})
        self.theta, self.gamma, self.r = res.x[:k], res.x[k:-1], float(np.exp(res.x[-1]))
        msg = res.message if isinstance(res.message, str) else res.message.decode()
        print(f"  nb fit: {res.nit} iters, {msg}, r {self.r:.2f}, nll/row {res.fun / len(y):.4f}")
        return self

    def block(self, name):
        return self.theta[self.off[name]: self.off[name] + self.n[name]]
