import json
from pathlib import Path

import numpy as np

from preprocess import WIDTH, forward, get_data_dir, load_data, load_weights
from train import unit, write_submission

OUT = Path("extra/v8")
SIN_A = 0.2
d = get_data_dir()
W = load_weights(d)
H5 = forward(W, load_data(d)[0])[1][5]
z7 = np.load("extra/v7/local_run/extra/train_v7.npz")
w7 = W["fc6.weight"] - np.outer(z7["v"], z7["d"])
lb = np.load("data/external/lb_files.npz")
b5 = np.load("extra/v5/local_run/extra/train_v5.npz")
b6 = np.load("extra/v6/local_run/extra/train_v6.npz")
dirs = {"d_perp": lb["d_perp"], "V3": lb["d_v3"], "V4": lb["d_v4"], "V5": b5["d"], "V6": b6["d"], "d0": b6["d0"], "V7": z7["direction"]}
reads = json.load(open(OUT / "readings.json"))
probes = np.load(OUT / "probes.npz") if (OUT / "probes.npz").exists() else None
if probes is not None:
    dirs.update({f"e{k + 1}": probes["P"][k] for k in range(len(probes["P"]))})
names = [n for n in dirs if n in reads]
D = np.array([unit(dirs[n]) for n in names])
D *= np.sign(D @ unit(z7["direction"]))[:, None]
r = np.array([reads[n] for n in names])
live = H5.max(0) > 0
ev, E = np.linalg.eigh(H5[:, live].T @ H5[:, live] / len(H5))
N = np.zeros((WIDTH, int((ev >= 1e-9).sum())))
N[live] = E[:, ev >= 1e-9]
A = D @ N
x = np.linalg.lstsq(A, r, rcond=None)[0]
t = N @ x
print(f"{len(names)} readings, rank {np.linalg.matrix_rank(A, tol=1e-8)} of {N.shape[1]} | |t| {np.linalg.norm(t):.5f} -> Part 2 >= {100 * np.linalg.norm(t) - 50:.3f}"
      f" | fit residual {np.abs(A @ x - r).max():.1e}")
write_submission(OUT / "submission.csv", w7, t)
if probes is None:
    Q = np.linalg.svd(A)[2][np.linalg.matrix_rank(A, tol=1e-8):]
    P = np.array([np.sqrt(1 - SIN_A ** 2) * unit(t) + SIN_A * (N @ q) for q in Q])
    np.savez(OUT / "probes.npz", P=P, t0=t)
    for k, p in enumerate(P):
        write_submission(OUT / "probes" / f"v8_probe_e{k + 1}.csv", w7, p)
    print(f"wrote {len(P)} probes (expected LB each {100 * np.sqrt(1 - SIN_A ** 2) * np.linalg.norm(t):.2f} + {100 * SIN_A:.0f} x the truth's coordinate e_k)")
