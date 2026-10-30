import os
from pathlib import Path

import numpy as np
import pandas as pd

TRAIN_FILE, TEST_FILE = "train.csv", "test.csv"
TARGET = "quality_score"
DIAM = [f"diameter_{i:02d}" for i in range(1, 21)]
GAP4 = (2, 6, 11, 16)


def find_data():
    for dirpath, _, files in os.walk("/kaggle/input"):
        if TRAIN_FILE in files and TEST_FILE in files:
            if "diameter_01" in pd.read_csv(Path(dirpath) / TRAIN_FILE, nrows=1).columns:
                return Path(dirpath)
    try:
        return Path(__file__).resolve().parent.parent / "data" / "raw"
    except NameError:
        return Path.cwd() / "data" / "raw"


def load():
    d = find_data()
    tr, te = pd.read_csv(d / TRAIN_FILE), pd.read_csv(d / TEST_FILE)
    assert len(tr) == 6000 and len(te) == 4000 and TARGET in tr and TARGET not in te
    assert np.isfinite(tr[DIAM + [TARGET]].to_numpy()).all() and np.isfinite(te[DIAM].to_numpy()).all()
    print(f"data {d}: train {len(tr)}, test {len(te)}")
    return tr, te


def features(df):
    S = np.sort(df[DIAM].to_numpy(), axis=1)
    return S, np.diff(S, axis=1), S.std(axis=1)
