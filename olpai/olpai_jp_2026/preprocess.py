from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

T = 40
N_FOLDS = 5


def data_dir():
    for d in Path("/kaggle/input").glob("**/train.csv"):
        if "AUDp_l" in d.open().readline():
            return d.parent
    try:
        return Path(__file__).resolve().parent.parent / "data" / "raw"
    except NameError:
        return Path.cwd() / "data" / "raw"


def frames(df):
    df = df.assign(row=np.arange(len(df)), frame=df["id"].str.rsplit("_", n=1).str[1].astype(int))
    df = df.sort_values(["sample_id", "frame"], ignore_index=True)
    assert (df["frame"] == df.groupby("sample_id").cumcount()).all(), "frames are not 0..n-1 per sample"
    return df


def to_windows(df, values):
    n = df.groupby("sample_id", sort=True).size().to_numpy()
    assert n.max() <= T
    w, t = np.repeat(np.arange(len(n)), n), df["frame"].to_numpy()
    X = np.zeros((len(n), T, values.shape[1]), np.float32)
    X[w, t] = values
    return X, n, w, t


def tdiff(X, mask):
    d = np.zeros_like(X)
    d[:, 1:] = X[:, 1:] - X[:, :-1]
    return d * mask[..., None]


def windows(df, brain, stats, mice, mouse_days):
    t, nf = df["frame"].to_numpy(), df.groupby("sample_id")["frame"].transform("size").to_numpy()
    meta = np.column_stack([t / (T - 1), nf / T, (nf - 1 - t) / (T - 1), df["day_n"].to_numpy() / 15,
                            (df["time"].to_numpy() - stats["tmu"]) / stats["tsd"]])
    mz = ((df[brain] - stats["mouse_mu"].loc[df["mouse_id"]].to_numpy())
          / stats["mouse_sd"].loc[df["mouse_id"]].to_numpy()).to_numpy()
    g = mz.mean(1, keepdims=True)
    Xb, n, w, t = to_windows(df, np.column_stack([mz - g, g]))
    mask = (np.arange(T)[None] < n[:, None]).astype(np.float32)
    d1 = tdiff(Xb, mask)
    d2 = tdiff(d1, mask * (np.arange(T)[None] >= 2))
    Xm, _, _, _ = to_windows(df, meta)
    X = np.concatenate([Xb, d1, d2, Xm, mask[..., None]], -1).astype(np.float32)
    first = np.concatenate([[0], np.cumsum(n)[:-1]])
    mouse = df["mouse_id"].to_numpy()[first]
    md = pd.MultiIndex.from_arrays([mouse, df["day_n"].to_numpy()[first]])
    return dict(X=X, n=n.astype(np.int64), m=np.searchsorted(mice, mouse).astype(np.int64),
                md=mouse_days.get_indexer(md).astype(np.int64), w=w, t=t)


def load():
    d = data_dir()
    tr, te = frames(pd.read_csv(d / "train.csv")), frames(pd.read_csv(d / "test.csv"))
    brain = [c for c in te.columns if c.endswith(("_l", "_r"))]
    assert len(brain) == 44 and "lever" in tr and "lever" not in te
    return tr, te, brain


def arrays(tr, te, brain):
    both = pd.concat([tr[["mouse_id", "day_n"] + brain], te[["mouse_id", "day_n"] + brain]])
    stats = {"tmu": tr["time"].mean(), "tsd": tr["time"].std(),
             "mouse_mu": both.groupby("mouse_id")[brain].mean(), "mouse_sd": both.groupby("mouse_id")[brain].std() + 1e-6}
    mice = np.sort(tr["mouse_id"].unique())
    assert set(te["mouse_id"]) <= set(mice)
    mouse_days = pd.MultiIndex.from_frame(both[["mouse_id", "day_n"]].drop_duplicates().sort_values(["mouse_id", "day_n"]))
    a, b = windows(tr, brain, stats, mice, mouse_days), windows(te, brain, stats, mice, mouse_days)
    ytr = np.zeros((len(a["n"]), T), np.float32)
    ytr[a["w"], a["t"]] = tr["lever"].to_numpy()
    fold = np.zeros(len(a["n"]), np.int64)
    for k, (_, idx) in enumerate(KFold(N_FOLDS, shuffle=True, random_state=42).split(a["n"])):
        fold[idx] = k
    out = {f"{key}tr": v for key, v in a.items()} | {f"{key}te": v for key, v in b.items()}
    return out | dict(ytr=ytr, fold=fold, n_mice=len(mice), n_md=len(mouse_days))
