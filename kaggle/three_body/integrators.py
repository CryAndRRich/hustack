import numpy as np

THETA = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
DKD = [("D", 0.5), ("K", 1.0), ("D", 0.5)]
KDK = [("K", 0.5), ("D", 1.0), ("K", 0.5)]
_XI, _LAM, _CHI = 0.1786178958448091, -0.2123418310626054, -0.06626458266981849


def compose(base, weights):
    return [(kind, c * w) for w in weights for kind, c in base]


def merge(ops):
    out = []
    for kind, c in ops:
        if out and out[-1][0] == kind:
            out[-1] = (kind, out[-1][1] + c)
        else:
            out.append((kind, c))
    return out


def symmetric(half, middle):
    return half + [middle] + half[::-1]


def forest_ruth(base, theta):
    return compose(base, [theta, 1.0 - 2.0 * theta, theta])


PEFRL_D = symmetric([("D", _XI), ("K", (1 - 2 * _LAM) / 2), ("D", _CHI), ("K", _LAM)], ("D", 1 - 2 * (_CHI + _XI)))
PEFRL_K = [("K" if kind == "D" else "D", c) for kind, c in PEFRL_D]
RK4 = ([[], [0.5], [0.0, 0.5], [0.0, 0.0, 1.0]], [1 / 6, 1 / 3, 1 / 3, 1 / 6])
RK38 = ([[], [1 / 3], [-1 / 3, 1.0], [1.0, -1.0, 1.0]], [1 / 8, 3 / 8, 3 / 8, 1 / 8])


CANDIDATES = {
    "fr_DKD": ("split", forest_ruth(DKD, THETA), "pow15"),
    "fr_DKD_merged": ("split", merge(forest_ruth(DKD, THETA)), "pow15"),
    "fr_DKD_theta15": ("split", forest_ruth(DKD, 1.35120719195966), "pow15"),
    "fr_DKD_theta4": ("split", forest_ruth(DKD, 1.3512), "pow15"),
    "fr_DKD_rrr": ("split", forest_ruth(DKD, THETA), "rrr"),
    "fr_DKD_pow3": ("split", forest_ruth(DKD, THETA), "pow3"),
    "fr_KDK": ("split", forest_ruth(KDK, THETA), "pow15"),
    "pefrl_D": ("split", PEFRL_D, "pow15"),
    "pefrl_K": ("split", PEFRL_K, "pow15"),
    "rk4": ("rk", RK4, "pow15"),
    "rk38": ("rk", RK38, "pow15"),
}

ARITH_VARIANTS = {"fr_DKD": ["fr_DKD", "fr_DKD_merged", "fr_DKD_rrr", "fr_DKD_pow3"]}

for _name, (_kind, _ops, _) in CANDIDATES.items():
    if _kind == "split":
        for _k in "DK":
            assert abs(sum(c for k, c in _ops if k == _k) - 1.0) < 1e-12, f"{_name}: {_k} coefficients do not sum to 1"
    else:
        assert abs(sum(_ops[1]) - 1.0) < 1e-15, f"{_name}: weights do not sum to 1"


def acc(p, m, inv3):
    d01, d02, d12 = p[:, 1] - p[:, 0], p[:, 2] - p[:, 0], p[:, 2] - p[:, 1]
    r2 = [np.einsum("ij,ij->i", d, d)[:, None] for d in (d01, d02, d12)]
    if inv3 == "pow15":
        i01, i02, i12 = (x ** -1.5 for x in r2)
    elif inv3 == "rrr":
        r = [np.sqrt(x) for x in r2]
        i01, i02, i12 = (1.0 / (x * x * x) for x in r)
    else:
        i01, i02, i12 = (np.sqrt(x) ** -3 for x in r2)
    a = np.empty_like(p)
    a[:, 0] = m[:, 1:2] * i01 * d01 + m[:, 2:3] * i02 * d02
    a[:, 1] = -m[:, 0:1] * i01 * d01 + m[:, 2:3] * i12 * d12
    a[:, 2] = -m[:, 0:1] * i02 * d02 - m[:, 1:2] * i12 * d12
    return a, np.minimum(np.minimum(r2[0], r2[1]), r2[2])[:, 0]


def run_split(state, ops, n_steps, inv3):
    m, p, v, t = state
    dt = (t / n_steps)[:, None, None]
    p, v, a = p.copy(), v.copy(), None
    r2min = np.full(len(t), np.inf)
    for _ in range(n_steps):
        for kind, c in ops:
            if kind == "D":
                p = p + (c * dt) * v
                a = None
            else:
                if a is None:
                    a, r2 = acc(p, m, inv3)
                    r2min = np.minimum(r2min, r2)
                v = v + (c * dt) * a
    return np.concatenate([p.reshape(-1, 9), v.reshape(-1, 9)], axis=1), np.sqrt(r2min)


def run_rk(state, tableau, n_steps, inv3):
    m, p, v, t = state
    A, b = tableau
    dt = (t / n_steps)[:, None, None]
    p, v = p.copy(), v.copy()
    r2min = np.full(len(t), np.inf)
    for _ in range(n_steps):
        kp, kv = [], []
        for i in range(len(b)):
            pi = p + dt * sum(A[i][j] * kp[j] for j in range(i)) if i else p
            vi = v + dt * sum(A[i][j] * kv[j] for j in range(i)) if i else v
            a, r2 = acc(pi, m, inv3)
            r2min = np.minimum(r2min, r2)
            kp.append(vi)
            kv.append(a)
        p = p + dt * sum(bi * k for bi, k in zip(b, kp))
        v = v + dt * sum(bi * k for bi, k in zip(b, kv))
    return np.concatenate([p.reshape(-1, 9), v.reshape(-1, 9)], axis=1), np.sqrt(r2min)


def predict(cand, state, n_steps):
    kind, spec, inv3 = CANDIDATES[cand]
    run = run_split if kind == "split" else run_rk
    return run(state, spec, int(n_steps), inv3)
