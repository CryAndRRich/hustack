from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

N_ITEMS = 9724
TRAIN_SHARE = 0.8
LIKE = 4.0


def find(name):
    root = Path("/kaggle/input")
    hits = sorted(root.rglob(name)) if root.exists() else []
    if hits:
        return hits[0].parent
    try:
        return Path(__file__).resolve().parent.parent / "data"
    except NameError:
        return Path.cwd() / "data"


def load():
    d = find("sample_submission.csv")
    tr, te = pd.read_csv(d / "train.csv"), pd.read_csv(d / "test.csv")
    times = pd.read_csv(find("ext_times.csv") / "ext_times.csv")
    n0 = len(tr)
    tr = tr.merge(times, on=["userId", "time_idx"], how="left", validate="one_to_one")
    tr = tr.sort_values(["userId", "time_idx"]).reset_index(drop=True)
    assert len(tr) == n0 and tr.movieId.max() < N_ITEMS and not tr.duplicated(["userId", "movieId"]).any()
    assert te.userId.isin(tr.userId).all() and te.id.is_unique
    assert (tr.dropna(subset=["timestamp"]).groupby("userId").timestamp.diff().dropna() >= 0).all()
    n_users = tr.userId.max() + 1
    print(f"train {len(tr)} rows, {n_users} users, {tr.movieId.nunique()} items; test {len(te)} users; "
          f"timestamps for {tr.timestamp.notna().mean():.4f} of rows")
    return tr, te, n_users


def matrix(df, n_users, tau=np.inf):
    age = df.groupby("userId").time_idx.transform("max") - df.time_idx
    w = np.exp(-age.to_numpy() / tau).astype(np.float32)
    return sp.csr_matrix((w, (df.userId, df.movieId)), shape=(n_users, N_ITEMS))


def split(tr, q):
    n = tr.groupby("userId").time_idx.transform("size")
    cut = np.round(n * q)
    early = tr.time_idx < cut
    late = tr[~early & (tr.time_idx < cut + np.round(cut * (1 - TRAIN_SHARE) / TRAIN_SHARE)) & (tr.rating >= LIKE)]
    targets = late.groupby("userId").movieId.agg(list).to_dict()
    print(f"cut {q}: input {early.sum()} rows, {len(late)} liked targets over {len(targets)} users")
    return tr[early], targets
