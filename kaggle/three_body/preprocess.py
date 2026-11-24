import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp

POS0 = [f"{a}{i}_0" for i in (1, 2, 3) for a in "xyz"]
VEL0 = [f"v{a}{i}_0" for i in (1, 2, 3) for a in "xyz"]
TARGETS = [f"{a}{i}_t" for i in (1, 2, 3) for a in "xyz"] + [f"v{a}{i}_t" for i in (1, 2, 3) for a in "xyz"]
MASS = ["m1", "m2", "m3"]
TEST_LIKE_RMIN0 = 2.3


def find_data():
    roots = [Path("/kaggle/input")]
    try:
        roots.append(Path(__file__).resolve().parent.parent / "data")
    except NameError:
        pass
    roots.append(Path.cwd() / "data")
    for root in roots:
        if not root.exists():
            continue
        for dirpath, _, files in os.walk(root, followlinks=True):
            if "train.csv" in files and "test.csv" in files:
                tr = pd.read_csv(Path(dirpath) / "train.csv", nrows=1)
                if "t_target" in tr.columns and "vz3_t" in tr.columns:
                    return Path(dirpath)
    raise FileNotFoundError("train.csv/test.csv with t_target not found; attach the competition data")


def initial(df):
    return (df[MASS].to_numpy(float), df[POS0].to_numpy(float).reshape(-1, 3, 3),
            df[VEL0].to_numpy(float).reshape(-1, 3, 3), df["t_target"].to_numpy(float))


def subset(state, idx):
    return tuple(x[idx] for x in state)


def stack(a, b):
    return tuple(np.concatenate([x, y]) for x, y in zip(a, b))


def rmin0(state):
    p = state[1]
    return np.min([np.linalg.norm(p[:, j] - p[:, i], axis=1) for i, j in [(0, 1), (0, 2), (1, 2)]], axis=0)


def rhs(t, y, m):
    p0, p1, p2 = y[0:3], y[3:6], y[6:9]
    d01, d02, d12 = p1 - p0, p2 - p0, p2 - p1
    i01 = (d01 @ d01) ** -1.5
    i02 = (d02 @ d02) ** -1.5
    i12 = (d12 @ d12) ** -1.5
    a0 = m[1] * i01 * d01 + m[2] * i02 * d02
    a1 = -m[0] * i01 * d01 + m[2] * i12 * d12
    a2 = -m[0] * i02 * d02 - m[1] * i12 * d12
    return np.concatenate([y[9:], a0, a1, a2])


def run_ref(state, tol):
    m, p, v, t = state
    out = np.full((len(t), 18), np.nan)
    for k in range(len(t)):
        y0 = np.concatenate([p[k].ravel(), v[k].ravel()])
        sol = solve_ivp(rhs, (0.0, t[k]), y0, method="DOP853", rtol=tol, atol=tol, args=(m[k],))
        if sol.success:
            out[k] = sol.y[:, -1]
    return out


def row_rms(a, b):
    return np.sqrt(np.mean((a - b) ** 2, axis=1))


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def energy_angmom(m, p, v):
    ke = 0.5 * np.einsum("ni,nij,nij->n", m, v, v)
    pe = -sum(m[:, i] * m[:, j] / np.linalg.norm(p[:, j] - p[:, i], axis=1) for i, j in [(0, 1), (0, 2), (1, 2)])
    return ke + pe, np.einsum("ni,nij->nj", m, np.cross(p, v))
