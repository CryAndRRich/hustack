import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
N1, N2 = 50, 200


def load():
    d = np.loadtxt(ROOT / "data" / "Mobserved50by200.csv", delimiter=",", skiprows=2)
    k = d[:, 0].astype(int) - 1
    M = np.full((N1, N2), np.nan)
    M[k % N1, k // N1] = d[:, 1]
    return M


def als(M, r, lam=1e-9, iters=500, seed=0):
    mask = ~np.isnan(M)
    X = np.where(mask, M, 0.0)
    rng = np.random.default_rng(seed)
    U, V = rng.standard_normal((N1, r)), rng.standard_normal((N2, r))
    I = lam * np.eye(r)
    for _ in range(iters):
        for i in range(N1):
            Vi = V[mask[i]]
            U[i] = np.linalg.solve(Vi.T @ Vi + I, Vi.T @ X[i, mask[i]])
        for j in range(N2):
            Uj = U[mask[:, j]]
            V[j] = np.linalg.solve(Uj.T @ Uj + I, Uj.T @ X[mask[:, j], j])
    return U @ V.T


def pick_rank(M, ranks, frac=0.1, seed=0):
    obs = np.argwhere(~np.isnan(M))
    held = obs[np.random.default_rng(seed).random(len(obs)) < frac]
    Mtr = M.copy()
    Mtr[held[:, 0], held[:, 1]] = np.nan
    out = {}
    for r in ranks:
        P = als(Mtr, r)
        out[r] = np.sqrt(np.mean((P[held[:, 0], held[:, 1]] - M[held[:, 0], held[:, 1]]) ** 2))
        print(f"rank {r:2d}  holdout rmse {out[r]:.3e}")
    return min(out, key=out.get)


def write(Mhat, path):
    with open(path, "w") as fh:
        fh.write(f"IndexValues,MatrixValues\n{-N1},{N2}\n")
        for k, v in enumerate(Mhat.flatten(order="F")):
            fh.write(f"{k + 1},{float(v)!r}\n")


if __name__ == "__main__":
    M = load()
    r = pick_rank(M, range(1, 13))
    P = als(M, r)
    mask = ~np.isnan(M)
    print(f"rank {r}: train rmse {np.sqrt(np.mean((P[mask] - M[mask]) ** 2)):.3e}")
    print("singular values", np.round(np.linalg.svd(P, compute_uv=False)[: r + 2], 4))
    Mhat = np.where(mask, M, P)
    out = Path("/kaggle/working") if Path("/kaggle/working").exists() else ROOT
    write(Mhat, out / "submission.csv")
    print(f"SUBMIT submission.csv  rank={r}")
