import itertools
from pathlib import Path

import numpy as np
import pandas as pd

PARTS = ["train", "val", "test", "eval"]
VIS = 6


def data_dir():
    for p in Path("/kaggle/input").rglob("train_data.npy"):
        return p.parent
    try:
        return Path(__file__).resolve().parent.parent / "data"
    except NameError:
        return Path.cwd() / "data"


def load():
    d = data_dir()
    X = {p: np.load(d / f"{p}_data.npy", allow_pickle=False).astype(np.float32) for p in PARTS}
    y = {p: np.load(d / f"{p}_label.npy", allow_pickle=False).astype(int) for p in PARTS if p != "eval"}
    ids = pd.read_csv(d / "sample_submission.csv", encoding="utf-8-sig")["id"].to_numpy()
    assert all(x.shape[1:] == (14, 14, 11) for x in X.values()) and len(ids) == len(X["eval"])
    print("data", d, {p: X[p].shape[0] for p in PARTS}, "train classes", np.bincount(y["train"]))
    return X, y, ids


def features(x):
    n, b = len(x), x.shape[-1]
    px = x.reshape(n, -1, b)
    cols = {}
    for name, f in [("mean", np.mean), ("std", np.std), ("med", np.median)]:
        v = f(px, axis=1)
        cols.update({f"{name}_{i}": v[:, i] for i in range(b)})
    for q in (10, 90):
        v = np.percentile(px, q, axis=1)
        cols.update({f"p{q}_{i}": v[:, i] for i in range(b)})
    c = x[:, 5:9, 5:9].reshape(n, -1, b).mean(1)
    cols.update({f"ctr_{i}": c[:, i] - cols[f"mean_{i}"] for i in range(VIS)})
    mean = np.stack([cols[f"mean_{i}"] for i in range(VIS)], 1)
    tot = mean.sum(1, keepdims=True) + 1e-6
    cols.update({f"shape_{i}": mean[:, i] / tot[:, 0] for i in range(VIS)})
    for i, j in itertools.combinations(range(VIS), 2):
        cols[f"nd_{i}_{j}"] = (mean[:, i] - mean[:, j]) / (mean[:, i] + mean[:, j] + 1e-6)
        pnd = (px[..., i] - px[..., j]) / (np.abs(px[..., i]) + np.abs(px[..., j]) + 1e-6)
        cols[f"ndstd_{i}_{j}"] = pnd.std(1)
    return pd.DataFrame(cols).astype(np.float32)
