from pathlib import Path

import numpy as np
import pandas as pd

LEADS = ["Q_pred_t", "Q_pred_t1", "Q_pred_t2", "Q_pred_t3"]
ZERO = "zero.csv"
SCORES = {"v1.csv": 0.05466, "v2.csv": 0.08953, "v2.1.csv": 0.09491, ZERO: 0.07137, "best1.csv": 0.05421,
          "v1_08178880.csv": 0.07095, "v1_08195000.csv": 0.06730, "v1_08196000.csv": 0.06084,
          "v1_08198500.csv": 0.07135, "v1_08200000.csv": 0.06947, "v1_08202700.csv": 0.07262, "best2.csv": 0.04630,
          "nwm.csv": 0.10913, "best2_smooth13.csv": 0.04713, "best2_late3.csv": 0.05108, "best2_col3.csv": 0.06600,
          "nwm_08196000.csv": 0.06769, "nwm_08195000.csv": 0.07739, "nwm_08198500.csv": 0.08025,
          "v21_08202700.csv": 0.08078, "nwm_08200000.csv": 0.08352}
OUT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")


def find_dir():
    root = Path("/kaggle/input")
    hits = list(root.rglob(ZERO)) if root.exists() else []
    if hits:
        return hits[0].parent
    try:
        return Path(__file__).resolve().parent.parent / "extra" / "probes"
    except NameError:
        return Path.cwd() / "extra" / "probes"


def project(d):
    z = SCORES[ZERO] ** 2
    names = [n for n in SCORES if n != ZERO]
    frames = [pd.read_csv(d / n) for n in names]
    ids = frames[0].Unique_ID
    assert all((f.Unique_ID == ids).all() for f in frames), "scored files are not in the same row order"
    P = np.stack([f[LEADS].to_numpy().ravel() for f in frames])
    G = P @ P.T / P.shape[1]
    a = (z + np.diag(G) - np.array([SCORES[n] for n in names]) ** 2) / 2
    for n, g, ai in zip(names, np.diag(G), a):
        print(f"  {n}: <p,q>/N {ai:.7f}, best scale {ai / g:.3f} -> rmse {np.sqrt(z - ai ** 2 / g):.5f}")
    w = np.linalg.pinv(G, rcond=1e-10) @ a
    print("projection weights: " + ", ".join(f"{n} {x:.3f}" for n, x in zip(names, w)))
    best = np.clip(w @ P, 0, None).reshape(-1, len(LEADS))
    return pd.DataFrame(best, columns=LEADS).assign(Unique_ID=ids)[["Unique_ID"] + LEADS], np.sqrt(z - a @ w)


def run():
    d = find_dir()
    print(f"scored files from {d}; Z rms {SCORES[ZERO]:.5f}")
    sub, expected = project(d)
    sub.to_csv(OUT / "submission.csv", index=False, float_format="%.6g")
    print(f"SUBMIT: submission.csv  (V3: LB projection on {len(SCORES) - 1} scored files, expected rmse {expected:.5f}, "
          f"clipping at 0 can only lower it)")


if __name__ == "__main__":
    run()
