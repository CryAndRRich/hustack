from pathlib import Path

import pandas as pd

NUM = [f"x{i:02d}" for i in range(1, 16)]
CATS = {"cat_a": ["alpha", "beta", "gamma", "delta", "epsilon"], "cat_b": ["low", "mid", "high"]}
MISSING = {"x02", "x05", "x08", "x10", "x12"}


def data_dir():
    root = Path("/kaggle/input")
    for p in sorted(root.rglob("train.csv")) if root.exists() else []:
        if {"x01", "cat_a", "target"} <= set(pd.read_csv(p, nrows=0).columns):
            return p.parent
    try:
        return Path(__file__).resolve().parent.parent / "data"
    except NameError:
        return Path.cwd() / "data"


def out_dir():
    p = Path("/kaggle/working")
    return p if p.exists() else Path(".")


def load():
    d = data_dir()
    tr, te = pd.read_csv(d / "train.csv"), pd.read_csv(d / "test.csv")
    for df in (tr, te):
        assert set(df.columns[df.isna().any()]) <= MISSING, df.columns[df.isna().any()].tolist()
        for c, levels in CATS.items():
            assert set(df[c]) <= set(levels), c
    print(f"data {d}: train {tr.shape}, test {te.shape}")
    return tr, te
