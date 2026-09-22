import math

import numpy as np

N_ELLIPSES = 67
SHAPES = (("2:1", 2.0, 1.0), ("3:2", 3.0, 2.0), ("3:1", 3.0, 1.0))
SHAPE_AB = {k: (a, b) for k, a, b in SHAPES}

HEX67_AREA = 20.0 * (2.0 + 6.0 * math.sqrt(3.0))
HEX67_SCORE = (N_ELLIPSES * math.pi / HEX67_AREA * 1000.0) ** 2


def shape_matrix(theta, a, b):
    ct, st = np.cos(theta), np.sin(theta)
    a2, b2 = a * a, b * b
    return (a2 * ct * ct + b2 * st * st,
            (a2 - b2) * ct * st,
            a2 * st * st + b2 * ct * ct)


def half_extents(theta, a, b):
    A11, _, A22 = shape_matrix(theta, a, b)
    return np.sqrt(A11), np.sqrt(A22)


def _pw_F(lam, Ai, Aj, rx, ry):
    a11, a12, a22 = (np.asarray(t)[..., None] for t in Ai)
    b11, b12, b22 = (np.asarray(t)[..., None] for t in Aj)
    rx_, ry_ = np.asarray(rx)[..., None], np.asarray(ry)[..., None]
    m11 = a11 + lam * (b11 - a11)
    m12 = a12 + lam * (b12 - a12)
    m22 = a22 + lam * (b22 - a22)
    det = m11 * m22 - m12 * m12
    num = m22 * rx_ * rx_ - 2.0 * m12 * rx_ * ry_ + m11 * ry_ * ry_
    return lam * (1.0 - lam) * num / det


def contact_mu(Ai, Aj, rx, ry, coarse=33, refine=14):
    p = np.asarray(rx).shape[0]
    lam = np.linspace(0.0, 1.0, coarse + 2)[1:-1]
    vals = _pw_F(lam[None, :], Ai, Aj, rx, ry)
    best = vals.max(axis=1)
    k = vals.argmax(axis=1)
    step = lam[1] - lam[0]
    lo = np.clip(lam[k] - step, 1e-12, 1.0 - 1e-12)
    hi = np.clip(lam[k] + step, 1e-12, 1.0 - 1e-12)
    frac = np.array([0.25, 0.5, 0.75])
    rows = np.arange(p)
    for _ in range(refine):
        pts = lo[:, None] + (hi - lo)[:, None] * frac[None, :]
        vals = _pw_F(pts, Ai, Aj, rx, ry)
        best = np.maximum(best, vals.max(axis=1))
        c = pts[rows, vals.argmax(axis=1)]
        half = (hi - lo) * 0.25
        lo = np.clip(c - half, 1e-12, 1.0 - 1e-12)
        hi = np.clip(c + half, 1e-12, 1.0 - 1e-12)
    return np.sqrt(np.maximum(best, 0.0))


def pair_indices(n=N_ELLIPSES):
    return np.triu_indices(n, 1)


def pairwise_mu(xy, theta, a, b, coarse=33, refine=14):
    i, j = pair_indices(len(xy))
    A11, A12, A22 = shape_matrix(theta, a, b)
    return contact_mu((A11[i], A12[i], A22[i]), (A11[j], A12[j], A22[j]),
                      xy[j, 0] - xy[i, 0], xy[j, 1] - xy[i, 1], coarse, refine)


def density(a, b, W, H):
    return N_ELLIPSES * math.pi * a * b / (W * H)


def score_shape(a, b, W, H):
    return (density(a, b, W, H) * 1000.0) ** 2


def validate(a, b, W, H, xy, deg, coarse=33, refine=14):
    xy = np.asarray(xy, dtype=np.float64)
    theta = np.radians(np.asarray(deg, dtype=np.float64))
    hx, hy = half_extents(theta, a, b)
    slacks = np.concatenate([xy[:, 0] - hx, W - xy[:, 0] - hx,
                             xy[:, 1] - hy, H - xy[:, 1] - hy])
    mu = pairwise_mu(xy, theta, a, b, coarse, refine)
    return {
        "n": len(xy),
        "W": W, "H": H,
        "wall_slack": float(slacks.min()),
        "min_mu": float(mu.min()),
        "n_overlaps": int((mu < 1.0).sum()),
        "ok": bool(len(xy) == N_ELLIPSES and slacks.min() >= 0.0 and mu.min() >= 1.0),
        "density": density(a, b, W, H),
        "score": score_shape(a, b, W, H),
    }


def fmt(v):
    s = f"{v:.10g}"
    if "e" in s or "E" in s:
        s = f"{v:.10f}"
    return s


def roundtrip(v):
    return float(fmt(v))


def build_submission_text(packings):
    lines = []
    for key, _, _ in SHAPES:
        W, H, xy, deg = packings[key]
        lines.append(f"ratio: {key}")
        lines.append(f"{fmt(W)}*{fmt(H)}")
        for idx in range(len(xy)):
            lines.append(f"{idx:03d}: {fmt(xy[idx, 0])} {fmt(xy[idx, 1])} {fmt(deg[idx])}")
    return "\n".join(lines)


def parse_submission_text(text):
    out = {}
    for block in text.split("ratio: ")[1:]:
        rows = block.strip().split("\n")
        key = rows[0].strip()
        W, H = (float(t) for t in rows[1].split("*"))
        vals = np.array([[float(t) for t in r.split(":")[1].split()] for r in rows[2:]])
        out[key] = (W, H, vals[:, :2], vals[:, 2])
    return out


def write_submission(packings, path):
    import pandas as pd
    text = build_submission_text(packings)
    pd.DataFrame({"row_id": [0], "submission_text": [text]}).to_csv(path, index=False)
    return text


def symmetry_residual(xy, b):
    x, y = xy[:, 0], xy[:, 1]
    cx, cy = x.mean(), y.mean()
    rx, ry = 2.0 * cx - x, 2.0 * cy - y
    d2 = (rx[:, None] - x[None, :]) ** 2 + (ry[:, None] - y[None, :]) ** 2
    nn = d2.argmin(axis=1)
    resid = np.sqrt(d2[np.arange(len(x)), nn])
    med = float(np.median(resid))
    return {"median": med, "max": float(resid.max()), "median_over_b": med / b}


def certify(a, b, Wu, Hv, uv, theta, eps=1e-9, guard=1e-8, tries=16):
    uv = np.asarray(uv, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    x, y = uv[:, 0] * a, uv[:, 1] * b
    W0, H0 = Wu * a, Hv * b
    hx, hy = half_extents(theta, a, b)

    if x.min() <= 0 or y.min() <= 0 or (W0 - x).min() <= 0 or (H0 - y).min() <= 0:
        return None, None, None, None, {"ok": False, "reason": "centre outside container"}

    mu = pairwise_mu(uv * np.array([a, b]), theta, a, b)
    s = max(float((hx / x).max()), float((hx / (W0 - x)).max()),
            float((hy / y).max()), float((hy / (H0 - y)).max()),
            1.0 / float(mu.min()))
    s *= (1.0 + eps) * (1.0 + guard)

    deg = np.degrees(theta) % 180.0
    for _ in range(tries):
        W = roundtrip(W0 * s)
        H = roundtrip(H0 * s)
        xy = np.array([[roundtrip(x[i] * s), roundtrip(y[i] * s)] for i in range(len(x))])
        d = np.array([roundtrip(v) for v in deg])
        rep = validate(a, b, W, H, xy, d)
        if rep["ok"]:
            rep["scale"] = s
            return W, H, xy, d, rep
        s *= 1.0 + 10.0 * guard
    return None, None, None, None, {"ok": False, "reason": "certify did not converge", "last": rep}


def self_test(verbose=True):
    checks = []

    def chk(name, got, want, tol=1e-9):
        ok = abs(got - want) <= tol * max(1.0, abs(want))
        checks.append((name, ok, f"got {got!r}, want {want!r}"))

    one = (np.array([1.0]), np.array([0.0]), np.array([1.0]))
    for d in (1.5, 2.0, 3.7):
        chk(f"two unit circles d={d}: mu=d/2",
            float(contact_mu(one, one, np.array([d]), np.array([0.0]))[0]), d / 2.0, 1e-7)

    a, b = 3.0, 1.0
    A0 = shape_matrix(np.array([0.0]), a, b)
    for d in (4.0, 6.0, 9.0):
        chk(f"aligned {a}:{b} along x d={d}: mu=d/(2a)",
            float(contact_mu(A0, A0, np.array([d]), np.array([0.0]))[0]), d / (2 * a), 1e-7)
        chk(f"aligned {a}:{b} along y d={d}: mu=d/(2b)",
            float(contact_mu(A0, A0, np.array([0.0]), np.array([d]))[0]), d / (2 * b), 1e-7)

    A90 = shape_matrix(np.array([np.pi / 2]), a, b)
    chk("T-contact touches at d=a+b",
        float(contact_mu(A0, A90, np.array([a + b]), np.array([0.0]))[0]), 1.0, 1e-7)

    rng = np.random.default_rng(0)
    th = rng.uniform(0, np.pi, 2)
    r = rng.normal(size=2) * 4.0
    base = float(contact_mu(shape_matrix(th[:1], a, b), shape_matrix(th[1:], a, b),
                            r[:1], r[1:2])[0])
    phi = 0.7
    c, s_ = math.cos(phi), math.sin(phi)
    rot = np.array([c * r[0] - s_ * r[1], s_ * r[0] + c * r[1]])
    chk("rotation invariance",
        float(contact_mu(shape_matrix(th[:1] + phi, a, b), shape_matrix(th[1:] + phi, a, b),
                         rot[:1], rot[1:2])[0]), base, 1e-7)
    chk("homogeneous of degree 1 in r",
        float(contact_mu(shape_matrix(th[:1], a, b), shape_matrix(th[1:], a, b),
                         r[:1] * 2.5, r[1:2] * 2.5)[0]), base * 2.5, 1e-7)

    for v in (0.0, 1.0, 59.999999999999, 123.456789012345, 1e-9, 180.0):
        chk(f"fmt round trip {v}", roundtrip(roundtrip(v)), roundtrip(v), 0.0)

    bad = [c for c in checks if not c[1]]
    if verbose:
        for name, ok, detail in checks:
            print(f"  [{'ok' if ok else 'FAIL'}] {name}" + ("" if ok else f"  -- {detail}"))
    if bad:
        raise AssertionError(f"{len(bad)} geometry self-tests failed: {[c[0] for c in bad]}")
    print(f"geometry self-test: {len(checks)}/{len(checks)} passed")
    return True
