from pathlib import Path

import numpy as np
import pandas as pd

CLIP = 1200
SAW = 37
DAY_HOURS = range(9, 19)


def find(name, col):
    root = Path("/kaggle/input")
    for p in sorted(root.rglob(name)) if root.exists() else []:
        if col in pd.read_csv(p, nrows=0).columns:
            return p
    try:
        return Path(__file__).resolve().parent.parent / "data" / name
    except NameError:
        return Path.cwd() / "data" / name


def load():
    tr_path = find("train.csv", "load_mw")
    tr, te = pd.read_csv(tr_path), pd.read_csv(tr_path.parent / "test.csv")
    for df in (tr, te):
        assert (df.hour == df.id % 24).all() and (df.day_of_week == df.id // 24 % 7).all()
    print(f"data {tr_path.parent}: train {tr.shape}, test {te.shape}")
    return tr, te


def load_correction(te):
    path = find("probe_correction.csv", "corr")
    assert path.exists(), "attach the peak-load-probe dataset (data/peak-load-probe.zip)"
    pc = pd.read_csv(path)
    assert (pc.id.values == te.id.values).all()
    return pc


def design(df):
    we = df.day_of_week.values >= 5
    day = df.hour.isin(DAY_HOURS).values
    t = df.temperature_c.values - 20
    X = np.eye(24)[df.hour.values]
    cols = [X, t, t**2, df.is_holiday.values, we & day, we & ~day, df.id.values % SAW, df.humidity_pct.values]
    names = [f"h{h}" for h in range(24)] + ["t", "t2", "hol", "we_day", "we_night", "saw", "hum"]
    return np.column_stack(cols).astype(float), names


def features(df):
    f = df[["hour", "day_of_week", "is_holiday", "temperature_c", "humidity_pct"]].copy()
    f["weekend"] = (df.day_of_week >= 5).astype(int)
    return f
