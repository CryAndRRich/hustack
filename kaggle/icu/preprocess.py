import time
from pathlib import Path

import numpy as np
import pandas as pd

ID, TGT = "id", "death"
CATS = ["sex", "dzgroup", "dzclass", "income", "race", "ca", "dnr"]
PROGNOSES = ["surv2m", "surv6m", "prg2m", "prg6m"]
N_TRAIN, N_TEST, N_FEAT = 5463, 3642, 42
SPLINE_MIN_UNIQUE = 20

OUT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
WORK = Path("/tmp/work")
LINES = []
T0 = time.time()


def say(*parts):
    msg = f"[{time.time() - T0:7.1f}s] " + " ".join(str(p) for p in parts)
    print(msg, flush=True)
    LINES.append(msg)


def data_roots():
    roots = [Path("/kaggle/input")]
    try:
        roots.append(Path(__file__).resolve().parent.parent / "data")
    except NameError:
        roots.append(Path.cwd() / "data")
    return [r for r in roots if r.exists()]


def find_csv(pred):
    for root in data_roots():
        for p in sorted(root.rglob("*.csv")):
            if pred(list(pd.read_csv(p, nrows=0).columns)):
                return p
    raise FileNotFoundError(pred)


def load():
    tr = pd.read_csv(find_csv(lambda c: ID in c and TGT in c and len(c) > 2))
    te = pd.read_csv(find_csv(lambda c: ID in c and TGT not in c and len(c) > 2))
    assert len(tr) == N_TRAIN and len(te) == N_TEST, (len(tr), len(te))
    feats = [c for c in te.columns if c != ID]
    assert len(feats) == N_FEAT and set(CATS + PROGNOSES) <= set(feats), feats
    df = pd.concat([tr[feats], te[feats]], ignore_index=True)
    num = [c for c in feats if c not in CATS]

    raw = df[num].assign(n_missing=df.isna().sum(axis=1).astype(float),
                         **{f"logit_{c}": np.log(df[c].clip(1e-3, 1 - 1e-3) / (1 - df[c].clip(1e-3, 1 - 1e-3)))
                            for c in PROGNOSES})
    cats = df[CATS].fillna("<NA>").astype("category")
    nan_cols = [c for c in raw.columns if raw[c].isna().any()]
    flags = raw[nan_cols].isna().astype(float).add_prefix("na_")
    filled = raw.fillna(raw.median())

    lin = filled.copy()
    for c in lin.columns:
        if lin[c].min() >= 0 and lin[c].skew() > 2:
            lin[c] = np.log1p(lin[c])
    spline_cols = [c for c in lin.columns if lin[c].nunique() >= SPLINE_MIN_UNIQUE]

    data = {
        "y": tr[TGT].values.astype(int),
        "test_id": te[ID].values,
        "n": len(tr),
        "frame": pd.concat([raw, cats], axis=1),
        "dense": pd.concat([filled, flags, cats], axis=1),
        "lin": pd.concat([lin, flags, pd.get_dummies(cats, dtype=float)], axis=1),
        "spline_cols": spline_cols,
    }
    say("train", tr.shape, "test", te.shape, "death rate", round(data["y"].mean(), 4),
        "| frame", data["frame"].shape, "dense", data["dense"].shape, "lin", data["lin"].shape,
        "spline cols", len(spline_cols))
    return data
