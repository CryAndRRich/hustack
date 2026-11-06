import numpy as np
from scipy.optimize import minimize

from preprocess import LAYERS

def tail(W, layer, H, A, C):
    i = LAYERS.index(layer)
    hs, zs = [H], []
    h = H
    for name in LAYERS[i:]:
        w = W[name + ".weight"]
        z = h @ w.T + W[name + ".bias"]
        if name == layer:
            z = z - (H @ C) @ A.T
        zs.append(z)
        h = z if name == "out" else np.maximum(z, 0.0)
        hs.append(h)
    return h[:, 0], hs, zs

def span_basis(H):
    ev, V = np.linalg.eigh(H.T @ H)
    o = np.argsort(-ev)
    return ev[o] / ev.sum(), V[:, o]

def removal(W, layer, H, base, v, d):
    return float(np.mean(base - tail(W, layer, H, v[:, None], d[:, None])[0]))

def min_norm_inspan(W, layer, v, d0, B, conds, starts):
    cons = [{"type": "eq", "fun": lambda c, H=H, base=base, t=t: (removal(W, layer, H, base, v, d0 + B @ c) - t) * 1e3}
            for H, base, t in conds]
    best = None
    for c0 in starts:
        res = minimize(lambda c: c @ c, c0, jac=lambda c: 2 * c, constraints=cons, method="SLSQP",
                       options={"maxiter": 1000, "ftol": 1e-14})
        viol = max(abs(f["fun"](res.x)) for f in cons)
        if viol < 1e-6 and (best is None or res.fun < best[0].fun):
            best = (res, viol)
    if best is None:
        raise RuntimeError("no start met the water constraints")
    return best[0].x, best[1]
