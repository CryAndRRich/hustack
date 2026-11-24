import numpy as np
from scipy.optimize import differential_evolution, least_squares, minimize_scalar

MIN_GAP_STEPS = 4
PHASE_STEP_MAX = 0.3
N_STARTS = 8


def basis(t, a, b):
    x = a * t + b
    with np.errstate(divide="ignore", invalid="ignore"):
        f = x * np.sin(1.0 / x)
    return np.where(x == 0, 0.0, f)


def gamma_hat(t, y, a, b, w=None):
    f = basis(t, a, b)
    if w is not None:
        f, y = f[w], y[w]
    ff = float(f @ f)
    return float(f @ y) / ff if ff > 0 else 0.0


def rms(t, y, a, b, g):
    return float(np.sqrt(np.mean((y - g * basis(t, a, b)) ** 2)))


def zeros(t, y):
    s = np.sign(y)
    i = np.flatnonzero(s[:-1] * s[1:] < 0)
    tz = t[i] - y[i] * (t[i + 1] - t[i]) / (y[i + 1] - y[i])
    if len(tz) < 3:
        return tz, tz
    dt = t[1] - t[0]
    g = np.diff(tz)
    left = np.concatenate([[np.inf], g])
    right = np.concatenate([g, [np.inf]])
    return tz, tz[(left >= MIN_GAP_STEPS * dt) & (right >= MIN_GAP_STEPS * dt)]


def _recip_steps(tz, t0):
    out = []
    for side in (tz[tz > t0], tz[tz < t0]):
        if len(side) >= 2:
            out.append(np.diff(np.sort(1.0 / np.abs(side - t0))))
    return np.concatenate(out) if out else np.array([])


def _t0_score(tz, t0):
    d = _recip_steps(tz, t0)
    if len(d) < 2 or np.any(d <= 0):
        return np.inf
    return float(np.std(d) / np.mean(d))


def init_from_zeros(t, y, tz):
    lo, hi = t[0] - 2.0, t[-1] + 2.0
    grid = np.linspace(lo, hi, 8001)
    scores = np.array([_t0_score(tz, c) for c in grid])
    c = grid[int(np.argmin(scores))]
    step = grid[1] - grid[0]
    r = minimize_scalar(lambda v: _t0_score(tz, v), bounds=(c - step, c + step), method="bounded",
                        options={"xatol": 1e-12})
    t0 = float(r.x)
    alpha = float(np.median(_recip_steps(tz, t0)) / np.pi)
    return t0, alpha, float(r.fun)


def _fit_ls(t, y, w, a, b, g):
    tw, yw = t[w], y[w]

    def res(p):
        return p[2] * basis(tw, p[0], p[1]) - yw

    def jac(p):
        x = p[0] * tw + p[1]
        s, c = np.sin(1.0 / x), np.cos(1.0 / x)
        dfx = p[2] * (s - c / x)
        return np.column_stack([dfx * tw, dfx, x * s])

    r = least_squares(res, [a, b, g], jac=jac, method="lm", xtol=1e-15, ftol=1e-15, gtol=1e-15, max_nfev=2000)
    return r.x


def resolved_mask(t, a, b):
    x = a * t + b
    dt = t[1] - t[0]
    return np.abs(a) * dt < PHASE_STEP_MAX * x * x


def refine(t, y, a, b):
    w = resolved_mask(t, a, b)
    g = gamma_hat(t, y, a, b, w)
    a, b, g = _fit_ls(t, y, w, a, b, g)
    a2, b2, g2 = _fit_ls(t, y, np.ones(len(t), bool), a, b, g)
    r2 = rms(t, y, a2, b2, g2)
    r1 = rms(t, y, a, b, g)
    return (a2, b2, g2, r2) if np.isfinite(r2) and r2 <= r1 else (a, b, g, r1)


def canonical(a, b, g):
    return (a, b, g) if a >= 0 else (-a, -b, g)


def fit_signal(t, y, seed):
    tz_all, tz = zeros(t, y)
    info = {"n_zeros": len(tz_all), "n_zeros_resolved": len(tz)}
    starts = []
    if len(tz) >= 3:
        t0, a0, sc = init_from_zeros(t, y, tz)
        info.update({"t0_init": t0, "alpha_init": a0, "init_score": sc})
        rng = np.random.default_rng(seed)
        starts = [(a0, t0)] + [(a0 * (1 + 0.02 * rng.standard_normal()), t0 + 0.002 * rng.standard_normal())
                               for _ in range(N_STARTS - 1)]
    best = None
    for a, t0 in starts:
        try:
            cand = refine(t, y, a, -a * t0)
        except (ValueError, np.linalg.LinAlgError):
            continue
        if np.isfinite(cand[3]) and (best is None or cand[3] < best[3]):
            best = cand
    info["fallback"] = best is None or best[3] > 1e-6 * max(float(np.abs(y).max()), 1e-12)
    if info["fallback"]:
        def obj(p):
            a, t0 = p
            g = gamma_hat(t, y, a, -a * t0)
            return rms(t, y, a, -a * t0, g)
        r = differential_evolution(obj, [(0.05, 20.0), (t[0] - 2.0, t[-1] + 2.0)], seed=seed, tol=1e-12,
                                   maxiter=400, popsize=40, polish=True)
        a, t0 = r.x
        cand = refine(t, y, a, -a * t0)
        if best is None or cand[3] < best[3]:
            best = cand
    a, b, g = canonical(*best[:3])
    info.update({"alpha": a, "beta": b, "gamma": g, "rms": best[3], "t0": -b / a if a else np.nan,
                 "y_absmax": float(np.abs(y).max())})
    return info
