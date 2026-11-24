import time
from pathlib import Path

import numpy as np
import pandas as pd

ID, GRP, TGT = "battle_id", "player_id", "battle_result"
CATS = ["character_path", "character_element", "battle_mode"]
N_TRAIN, N_TEST, N_PRED = 10000, 2500, 37
CYCLE_CUT, COUNT_CUT = 8, 4

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


def canonicalise(df):
    for c in CATS:
        low = df[c].str.strip().str.lower()
        top = df[c].groupby(low).agg(lambda s: s.value_counts().index[0])
        df[c] = low.map(top)
    return df


def numeric_matrix(X):
    X = pd.get_dummies(X, columns=[c for c in X.columns if X[c].dtype.name == "category"], dtype=float)
    nan_cols = [c for c in X.columns if X[c].isna().any()]
    X = pd.concat([X.fillna(X.median()), X[nan_cols].isna().astype(float).add_prefix("na_")], axis=1)
    return X.values.astype(np.float32)


def load():
    tr = pd.read_csv(find_csv(lambda c: ID in c and TGT in c and len(c) > 2))
    te = pd.read_csv(find_csv(lambda c: ID in c and TGT not in c and len(c) > 2))
    assert len(tr) == N_TRAIN and len(te) == N_TEST, (len(tr), len(te))
    num = [c for c in te.columns if c not in (ID, GRP, *CATS)]
    assert len(num) + len(CATS) == N_PRED, len(num)
    df = canonicalise(pd.concat([tr, te], ignore_index=True))
    for c in CATS:
        assert df[c].nunique() == 7 - 3 * (c == "battle_mode"), (c, df[c].unique())
    base = pd.concat([df[num], df[CATS].astype("category")], axis=1)
    plus = base.assign(cycle_late=(df.cycle_number >= CYCLE_CUT).astype(float),
                       many_enemies=(df.enemy_count >= COUNT_CUT).astype(float))
    is_tr = np.r_[np.ones(len(tr), bool), np.zeros(len(te), bool)]
    data = {
        "y": tr[TGT].values.astype(int),
        "groups": tr[GRP].values,
        "test_id": te[ID].values,
        "is_tr": is_tr,
        "X_base": numeric_matrix(base),
        "X_plus": numeric_matrix(plus),
        "X_frame": base.assign(**{c: df[c].astype(str) for c in CATS}),
    }
    say("train", tr.shape, "test", te.shape, "base matrix", data["X_base"].shape, "plus", data["X_plus"].shape,
        "win rate", round(data["y"].mean(), 4))
    return data
