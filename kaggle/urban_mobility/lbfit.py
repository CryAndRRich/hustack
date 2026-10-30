import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize

d = sys.argv[1].rstrip("/") + "/"
ARMS = pd.read_csv(d + "releases/v1.2/extra/test_arms_v1.2.csv")
T = ARMS["trend"].values
dates = pd.to_datetime(ARMS["date"])
t = np.arange(30)
X = np.column_stack([np.ones(30), t] + [(dates.dt.dayofweek == k).astype(float) for k in range(1, 7)])
coef, *_ = np.linalg.lstsq(X, np.log(T), rcond=None)
eff = np.r_[0, coef[2:]]
eff -= eff.mean()
logp = eff[dates.dt.dayofweek]
tilt = (t - t.mean()) / 29

V1 = ARMS["chronos2_cov"].values
FILES = {
    "v1": V1, "v1x0.8": 0.8 * V1, "v1x1.2": 1.2 * V1, "v1.1": T,
    "level-0.08": d + "releases/v1.2/extra/probe_v1.1_level-0.08.csv", "level+0.08": d + "releases/v1.2/extra/probe_v1.1_level+0.08.csv",
    "tilt-0.2": d + "releases/v1.2/extra/probe_v1.1_tilt-0.2.csv", "tilt+0.2": d + "releases/v1.2/extra/probe_v1.1_tilt+0.2.csv",
    "prof0.5": d + "releases/v1.2/extra/probe_v1.1_prof0.5.csv",
    "v1.2": ARMS["trend_lb"].values,
}
LOGF = {k: np.log(v if isinstance(v, np.ndarray) else pd.read_csv(v)["target"].values) for k, v in FILES.items()}
assert np.allclose(LOGF["v1.1"], np.log(T))

gh_x, gh_w = np.polynomial.hermite_e.hermegauss(24)
gh_w = gh_w / gh_w.sum()


def expected(logf, mu_a, sigma):
    dev = (logf - mu_a)[..., None] - sigma[..., None, None] * gh_x
    return (200 * np.tanh(np.abs(dev) / 2) * gh_w).sum(-1).mean(-1)


NAMES = ("c", "b", "gamma", "sigma")


def mu_of(c, b, g):
    return np.log(T) + c + b * tilt + (g - 1) * logp


def rss_of(theta, scores):
    c, b, g, s = theta
    if not (0.01 <= s <= 0.6 and -1.5 <= b <= 1.5 and -1 <= g <= 3 and -0.6 <= c <= 0.4):
        return 1e9
    m = mu_of(c, b, g)
    return sum((expected(LOGF[k], m, np.array(s)) - v) ** 2 for k, v in scores.items())


def fit(scores):
    starts = [(c, b, g, s) for c in (-0.1, 0.0, 0.1) for b in (-0.4, 0.0, 0.4) for g in (0.5, 1.0, 1.5) for s in (0.1, 0.25)]
    runs = [minimize(rss_of, x0, args=(scores,), method="Nelder-Mead", options=dict(xatol=1e-4, fatol=1e-6, maxiter=4000))
            for x0 in starts]
    runs.sort(key=lambda r: r.fun)
    best = dict(zip(NAMES, runs[0].x), rss=runs[0].fun)
    near = [r.x for r in runs if r.fun <= runs[0].fun + 1.0]
    spread = {n: (round(float(min(x[i] for x in near)), 2), round(float(max(x[i] for x in near)), 2)) for i, n in enumerate(NAMES)}
    return best, spread


def report(scores):
    best, spread = fit(scores)
    print("best:", {k: round(float(v), 3) for k, v in best.items()})
    print("range within rss+1:", spread)
    m = mu_of(best["c"], best["b"], best["gamma"])
    for k, v in scores.items():
        print(f"  {k:<11} LB {v:7.3f}  fitted {expected(LOGF[k], m, np.array(best['sigma'])):7.3f}")
    print(f"  optimum forecast (median = exp(mu)): expected {expected(m, m, np.array(best['sigma'])):.3f}, "
          f"mean {np.exp(m).mean():.0f}, Dec2 {np.exp(m[0]):.0f}, Dec31 {np.exp(m[-1]):.0f}")
    for name, lo, hi, h in (("level", "level-0.08", "level+0.08", 0.08), ("tilt", "tilt-0.2", "tilt+0.2", 0.2)):
        if lo in scores and hi in scores:
            s0, sl, sh = scores["v1.1"], scores[lo], scores[hi]
            curv = sl + sh - 2 * s0
            print(f"  parabola {name}: optimum {h * (sl - sh) / (2 * curv):+.3f} (curvature {curv:.2f})")


KNOWN = {"v1": 26.050242, "v1x0.8": 17.9244, "v1x1.2": 42.360976, "v1.1": 15.861115,
         "level-0.08": 16.149312, "level+0.08": 18.191921, "tilt-0.2": 13.288291, "tilt+0.2": 19.531396,
         "prof0.5": 14.682261, "v1.2": 10.140518}

if len(sys.argv) > 2 and sys.argv[2] == "selftest":
    rng = np.random.default_rng(1)
    true = dict(c=-0.05, b=-0.3, g=0.6, s=0.15)
    m = np.log(T) + true["c"] + true["b"] * tilt + (true["g"] - 1) * logp
    a = np.exp(m + true["s"] * rng.standard_normal(30))
    pub = rng.choice(30, 15, replace=False)
    fake = {k: float(np.mean(200 * np.abs(np.exp(f) - a)[pub] / (np.exp(f) + a)[pub])) for k, f in LOGF.items()}
    print("selftest truth:", true)
    report(fake)
else:
    print("all LB scores:")
    report(KNOWN)
