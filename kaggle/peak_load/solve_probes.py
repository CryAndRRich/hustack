import json
import sys

import numpy as np
import pandas as pd

from build_probes import EXTRA, ROUNDS, base_of, components
from preprocess import CLIP, load


def main(rnd):
    cfg = ROUNDS[rnd]
    _, te = load()
    b = base_of(cfg)
    free = b < CLIP
    out = EXTRA / rnd
    scores = json.loads((out / "scores.json").read_text())
    g = components(te)
    D = np.column_stack([pd.read_csv(out / f"probe_{k}.csv").load_mw.values - b for k in cfg["probe"]]
                        + [np.where(free, g[k], 0.0) for k in cfg["solved"]])

    c = np.zeros(D.shape[1])
    for j, k in enumerate(cfg["probe"]):
        dd = np.mean(D[:, j] ** 2)
        c[j] = (scores[k] ** 2 - cfg["base_lb"] ** 2 - dd) / 2
        print(f"{k:9s} LB {scores[k]:.6f}  rms d {np.sqrt(dd):.3f}  alone: weight {-c[j] / dd:+.4f}, gain {c[j] ** 2 / dd:.4f}")
    gram = D.T @ D / len(D)
    beta = -np.linalg.solve(gram, c)
    gain = float(c @ np.linalg.solve(gram, c))
    print("joint weights", dict(zip(cfg["probe"] + cfg["solved"], beta.round(4).tolist())))
    print(f"gram condition {np.linalg.cond(gram):.1e}")
    print(f"expected LB {np.sqrt(cfg['base_lb'] ** 2 - gain):.4f} (from {cfg['base_lb']}), public rows, before clip effects")
    p = np.minimum(b + D @ beta, CLIP)
    pd.DataFrame({"id": te.id, "load_mw": p}).to_csv(out / "submission.csv", index=False)
    print(f"SUBMIT: {out / 'submission.csv'}")


if __name__ == "__main__":
    main(sys.argv[1])
