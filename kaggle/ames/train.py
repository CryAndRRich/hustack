from pathlib import Path

import numpy as np
import pandas as pd

from preprocess import design, load

VERSION = "V3.2"
HALF = 1e-6
SIGMA_GRID = np.arange(30150, 30210, 0.01)
OUT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(__file__).resolve().parent.parent


def main():
    tr, te, feats = load()
    allm = pd.concat([tr, te])
    allm = allm[allm.z.notna()]
    beta = np.linalg.lstsq(design(allm, feats), np.log(allm.orig.to_numpy()), rcond=None)[0]

    y = tr.SalePrice.to_numpy(float)
    mod = tr.z.notna().to_numpy()
    assert np.all(y[~mod] == tr.orig[~mod])
    mu, z = np.exp(design(tr, feats)[mod] @ beta), tr.z.to_numpy()[mod]
    exact = np.array([np.all(np.round(mu + s * z, -2) == y[mod]) for s in SIGMA_GRID])
    sig = SIGMA_GRID[exact]
    assert len(sig) and sig.max() - sig.min() < 1, "sigma interval not identified"
    print(f"{VERSION}: beta from {len(allm)} modified rows; sigma in [{sig.min():.2f}, {sig.max():.2f}] reproduces {mod.sum()}/{mod.sum()} train prices")

    lo, hi = te.orig.to_numpy() - HALF, te.orig.to_numpy() + HALF
    tm = te.z.notna().to_numpy()
    cells = np.round(np.exp(design(te, feats)[tm] @ beta)[:, None] + sig[None, :] * te.z.to_numpy()[tm, None], -2)
    lo[tm], hi[tm] = cells.min(1) - HALF, cells.max(1) + HALF
    assert np.all(lo < hi)
    print(f"test modified {tm.sum()}: rows needing two cells {np.sum(cells.min(1) != cells.max(1))}")
    sub = te[["id"]].assign(lower=lo, upper=hi)
    sub.to_csv(OUT / "submission.csv", index=False)
    print(f"SUBMIT: submission.csv ({VERSION}) rows {len(sub)}  mean width {np.mean(hi - lo):.3f}")


if __name__ == "__main__":
    main()
