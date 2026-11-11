import glob
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

VERSION = "V1"
LO, HI, STEP = -6.0, 6.0, 0.002
SIGMAS = np.logspace(-3, 0.5, 71)
SIM_DRAWS = 3
OUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."


def find(name):
    hits = sorted(glob.glob(f"/kaggle/input/**/{name}", recursive=True)) or sorted(glob.glob(f"data/{name}"))
    if not hits:
        sys.exit(f"{name} not found")
    return hits[0]


def run_model(model, X):
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32)).reshape(-1).numpy().astype(np.float64)


def grid_outputs(model, X, col, v):
    F = np.empty((len(X), len(v)))
    for i0 in range(0, len(X), 25):
        b = np.repeat(X[i0:i0 + 25, None, :], len(v), axis=1)
        b[:, :, col] = v
        F[i0:i0 + 25] = run_model(model, b.reshape(-1, X.shape[1])).reshape(len(b), len(v))
    return F


def log_evidence(F, y, sigma, logprior):
    lp = logprior[None, :] - 0.5 * ((F - y[:, None]) / sigma) ** 2
    m = lp.max(1, keepdims=True)
    return float((m[:, 0] + np.log(np.exp(lp - m).sum(1)) + np.log(STEP) - np.log(sigma) - 0.5 * np.log(2 * np.pi)).sum())


def grid_posterior(F, y, sigma, logprior, v):
    lp = logprior[None, :] - 0.5 * ((F - y[:, None]) / sigma) ** 2
    w = np.exp(lp - lp.max(1, keepdims=True))
    w /= w.sum(1, keepdims=True)
    mean = w @ v
    return mean, np.sqrt(np.maximum(w @ v ** 2 - mean ** 2, 0)), v[lp.argmax(1)]


def root_posterior(F, y, logprior, v):
    d = F - y[:, None]
    i, g = np.where(d[:, :-1] * d[:, 1:] < 0)
    r = v[g] + d[i, g] / (d[i, g] - d[i, g + 1]) * STEP
    lw = np.interp(r, v, logprior) - np.log(np.abs(F[i, g + 1] - F[i, g]) / STEP)
    top = np.full(len(F), -np.inf)
    np.maximum.at(top, i, lw)
    w = np.exp(lw - top[i])
    s0, s1, s2 = (np.bincount(i, weights=w * r ** k, minlength=len(F)) for k in (0, 1, 2))
    n_roots = np.bincount(i, minlength=len(F))
    has = n_roots > 0
    mean = np.where(has, s1 / np.where(has, s0, 1), v[np.abs(d).argmin(1)])
    sd = np.where(has, np.sqrt(np.maximum(s2 / np.where(has, s0, 1) - mean ** 2, 0)), 0.0)
    return mean, sd, n_roots


def estimate(F, y, logprior, v):
    ev = np.array([log_evidence(F, y, s, logprior) for s in SIGMAS])
    sigma = float(SIGMAS[ev.argmax()])
    post, post_sd, mode = grid_posterior(F, y, sigma, logprior, v)
    roots, roots_sd, n_roots = root_posterior(F, y, logprior, v)
    return dict(sigma=sigma, ev=ev, post=post, post_sd=post_sd, map=mode, roots=roots, roots_sd=roots_sd,
                n_roots=n_roots)


def main():
    t0 = time.time()
    print(f"VERSION {VERSION}", flush=True)
    df = pd.read_csv(find("data.csv"))
    cols = [f"x{k}" for k in range(10)]
    X, y = df[cols].values.astype(np.float64), df["y"].values.astype(np.float64)
    zero = [k for k in range(10) if np.all(X[:, k] == 0)]
    if len(zero) != 1:
        sys.exit(f"expected exactly one all-zero column, found {zero}")
    col = zero[0]
    known = np.delete(X, col, axis=1)
    m, s = float(known.mean()), float(known.std())
    model = torch.jit.load(find("model.pt")).eval()
    print(f"erased column x{col}; prior N({m:.3f}, {s:.3f}^2) from the other 9 columns; rows {len(X)}", flush=True)
    print(f"model with zeros: MSE(model, y) {np.mean((run_model(model, X) - y) ** 2):.4f}, var(y) {y.var():.4f}", flush=True)

    v = np.arange(LO, HI + STEP / 2, STEP)
    logprior = -0.5 * ((v - m) / s) ** 2
    F = grid_outputs(model, X, col, v)
    res = float(np.quantile(np.abs(np.diff(F, axis=1)), 0.99))
    print(f"grid {len(v)} points on [{LO}, {HI}] in {time.time() - t0:.0f}s; output range per row q50 "
          f"{np.median(F.max(1) - F.min(1)):.2f}; output step q99 {res:.4f}", flush=True)

    est = estimate(F, y, logprior, v)
    sigma, ev = est["sigma"], est["ev"]
    k = int(ev.argmax())
    print(f"sigma by marginal likelihood {sigma:.4f} (grid edge: {k in (0, len(SIGMAS) - 1)}); log-evidence around it "
          f"{np.round(ev[max(k - 3, 0):k + 4] - ev[k], 1).tolist()}", flush=True)
    n_r, c_r = np.unique(np.minimum(est["n_roots"], 5), return_counts=True)
    print(f"roots per row: {dict(zip(n_r.tolist(), c_r.tolist()))} (5 = 5+)", flush=True)
    fit = np.mean((run_model(model, np.where(np.arange(10) == col, est["post"][:, None], X)) - y) ** 2)
    print(f"MSE(model, y) at the posterior mean {fit:.4f}; expected score from the posterior: grid "
          f"{np.mean(est['post_sd'] ** 2):.4f}, roots {np.mean(est['roots_sd'] ** 2):.4f}", flush=True)

    rng = np.random.default_rng(0)
    sim = {name: [] for name in ("post", "map", "roots")}
    sim_sigma = []
    for _ in range(SIM_DRAWS):
        truth = np.clip(rng.normal(m, s, len(X)), LO, HI)
        Xs = X.copy()
        Xs[:, col] = truth
        ys = run_model(model, Xs) + rng.normal(0, sigma, len(X))
        e = estimate(F, ys, logprior, v)
        sim_sigma.append(e["sigma"])
        for name in sim:
            sim[name].append(np.mean((e[name] - truth) ** 2))
    print(f"simulation ({SIM_DRAWS} draws: truth from the prior, y = model + N(0, {sigma:.4f}^2)); recovered sigma "
          f"{np.round(sim_sigma, 4).tolist()}; MSE " + ", ".join(f"{n} {np.mean(r):.4f}" for n, r in sim.items()), flush=True)

    pick = min(sim, key=lambda n: np.mean(sim[n]))
    os.makedirs(os.path.join(OUT, "extra", VERSION), exist_ok=True)
    for name in sim:
        pd.DataFrame({"id": df["id"], "answer": est[name]}).to_csv(
            os.path.join(OUT, "extra", VERSION, f"submission_{name}.csv"), index=False, float_format="%.6f")
    np.savez(os.path.join(OUT, "extra", VERSION, "posterior.npz"), **{k2: est[k2] for k2 in est})
    path = os.path.join(OUT, "submission.csv")
    pd.DataFrame({"id": df["id"], "answer": est[pick]}).to_csv(path, index=False, float_format="%.6f")
    print(f"picked {pick} (lowest simulated MSE); answer mean {est[pick].mean():+.3f} sd {est[pick].std():.3f}; "
          f"total {time.time() - t0:.0f}s", flush=True)
    print(f"SUBMIT: {path}", flush=True)


if __name__ == "__main__":
    main()
