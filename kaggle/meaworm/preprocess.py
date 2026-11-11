from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "Attack_type"
CATEGORICAL = ["proto", "service"]


def find_data_dir():
    roots = [Path("/kaggle/input")]
    try:
        roots.append(Path(__file__).resolve().parent.parent / "data")
    except NameError:
        roots.append(Path.cwd() / "data")
    for root in roots:
        if not root.exists():
            continue
        for p in sorted(root.rglob("*.csv")):
            if TARGET in pd.read_csv(p, nrows=0).columns and "fwd_pkts_tot" in pd.read_csv(p, nrows=0).columns:
                return p.parent
    raise FileNotFoundError("no csv with an Attack_type column under /kaggle/input or data/")


def load():
    d = find_data_dir()
    tables = {p: pd.read_csv(p) for p in sorted(d.glob("*.csv"))}
    train = next(t for t in tables.values() if TARGET in t.columns and len(t.columns) > 80)
    feats = [c for c in train.columns if c not in ("id", TARGET)]
    test = next(t for t in tables.values() if TARGET not in t.columns and set(feats) <= set(t.columns))
    for c in CATEGORICAL:
        cats = sorted(set(train[c].astype(str)) | set(test[c].astype(str)))
        train[c] = pd.Categorical(train[c].astype(str), categories=cats)
        test[c] = pd.Categorical(test[c].astype(str), categories=cats)
    classes = sorted(train[TARGET].unique())
    y = train[TARGET].map({c: i for i, c in enumerate(classes)}).values.astype(np.int64)
    print(f"data {d}: train {train.shape}, test {test.shape}, {len(feats)} features, {len(classes)} classes", flush=True)
    return train, test, y, classes, feats
