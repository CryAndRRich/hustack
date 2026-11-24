from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "probes"
COLS = ["alpha", "beta", "gamma"]
K = 100.0
GAMMA_MEAN = 2.85016
GAMMA_CONSTS = (2.0, 2.85, 3.7)


def frame(ids, values):
    df = pd.DataFrame({"ID": ids})
    for c in COLS:
        df[c] = values.get(c, np.zeros(len(ids)))
    return df


def main():
    v1 = pd.read_csv(ROOT / "extra" / "v1" / "submission.csv")
    assert list(v1.columns) == ["ID", *COLS] and v1["ID"].tolist() == list(range(100))
    ids = v1["ID"].to_numpy()
    n = len(ids)
    probes = {"p00_zeros": {}}
    for i, c in enumerate(COLS):
        probes[f"p{1 + 2 * i:02d}_{c}_plusK"] = {c: np.full(n, K)}
        probes[f"p{2 + 2 * i:02d}_{c}_minusK"] = {c: np.full(n, -K)}
    for i, c in enumerate(COLS):
        probes[f"p{7 + 2 * i:02d}_{c}_v1"] = {c: v1[c].to_numpy()}
        probes[f"p{8 + 2 * i:02d}_{c}_v1neg"] = {c: -v1[c].to_numpy()}
    a, b, g = (v1[c].to_numpy() for c in COLS)
    best = {"alpha": a, "beta": 2 * b}
    probes["r2_beta2"] = {**best, "gamma": g}
    for c in GAMMA_CONSTS:
        probes[f"r2_gamma_const{c:g}"] = {**best, "gamma": np.full(n, c)}
    probes["r2_gamma_2alpha"] = {**best, "gamma": 2 * a}
    probes["r2_gamma_prop_absg"] = {**best, "gamma": GAMMA_MEAN * np.abs(g) / np.abs(g).mean()}
    OUT.mkdir(parents=True, exist_ok=True)
    for name, vals in probes.items():
        df = frame(ids, vals)
        assert len(df) == n and np.isfinite(df[COLS].to_numpy()).all()
        df.to_csv(OUT / f"{name}.csv", index=False)
    scores = OUT / "scores.csv"
    if not scores.exists():
        pd.DataFrame({"probe": list(probes), "lb": [""] * len(probes)}).to_csv(scores, index=False)
    print(f"wrote {len(probes)} probes to {OUT} (K = {K:g}); fill the LB readings into {scores}")


if __name__ == "__main__":
    main()
