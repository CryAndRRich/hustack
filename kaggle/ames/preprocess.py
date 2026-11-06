import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from rrng import RRNG

DATA = Path(sys.argv[1]) if len(sys.argv) > 1 and Path(sys.argv[1]).is_dir() else Path("/kaggle/input")
SEED, N_MOD, N_TRAIN = 10292026, 1865, 1000


def find(name):
    return next(DATA.rglob(name))


def design(df, feats):
    X = df[feats].copy()
    X["Street"] = (X.Street == "Pave").astype(float)
    X["Central.Air"] = (X["Central.Air"] == "Y").astype(float)
    return np.column_stack([np.ones(len(X)), X.to_numpy(float)])


def host_stream(n):
    r = RRNG(SEED)
    mod = r.sample(n, N_MOD)
    z = r.rnorm(N_MOD)
    r.unif()
    train_rows = r.sample(n, N_TRAIN)
    zr = np.full(n, np.nan)
    zr[mod] = z
    return zr, train_rows, np.setdiff1d(np.arange(n), train_rows)


def load():
    tr = pd.read_csv(find("train_ames.csv"))
    te = pd.read_csv(find("test_ames.csv"))
    og = pd.read_csv(find("AmesHousing.txt"), sep="\t")
    og.columns = [("X" + c if c[0].isdigit() else c) for c in (re.sub(r"\W", ".", c) for c in og.columns)]
    feats = [c for c in te.columns if c != "id"]
    cc = og.dropna(subset=feats).reset_index(drop=True)
    cc[feats] = cc[feats].astype(tr[feats].dtypes.to_dict())
    zr, train_rows, test_rows = host_stream(len(cc))
    for df, rows in [(tr, train_rows), (te, test_rows)]:
        assert len(df) == len(rows) and (cc.loc[rows, feats].reset_index(drop=True) == df[feats]).all().all()
        df["orig"] = cc.SalePrice.to_numpy(float)[rows]
        df["z"] = zr[rows]
    return tr, te, feats
