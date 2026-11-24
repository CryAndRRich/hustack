from pathlib import Path

import numpy as np
import pandas as pd

N_SIGNALS = 100
N_TIMES = 1000


def find_data():
    kaggle = Path("/kaggle/input")
    if kaggle.exists():
        hits = sorted(kaggle.rglob("signals.csv"))
        if hits:
            return hits[0].parent
    try:
        base = Path(__file__).resolve().parent.parent / "data"
    except NameError:
        base = Path.cwd() / "data"
    hits = sorted(base.rglob("signals.csv"))
    if not hits:
        raise FileNotFoundError("signals.csv not found under /kaggle/input or data/")
    return hits[0].parent


def load():
    root = find_data()
    df = pd.read_csv(root / "signals.csv")
    names = [f"signal_{i}" for i in range(N_SIGNALS)]
    assert list(df.columns) == ["time", *names], "unexpected columns"
    assert len(df) == N_TIMES and df.notna().all().all(), "unexpected rows or missing values"
    t = df["time"].to_numpy(float)
    assert np.all(np.diff(t) > 0), "time is not increasing"
    sample = pd.read_csv(root / "sample_sub.csv")
    assert list(sample.columns) == ["ID", "alpha", "beta", "gamma"] and sample["ID"].tolist() == list(range(N_SIGNALS))
    print(f"data {root}: {len(t)} times in [{t[0]}, {t[-1]}], {len(names)} signals")
    return {"t": t, "Y": df[names].to_numpy(float), "ids": sample["ID"].to_numpy()}
