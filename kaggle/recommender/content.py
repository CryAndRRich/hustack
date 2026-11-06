import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sentence_transformers import SentenceTransformer

from preprocess import LIKE, N_ITEMS, find

TEXT_MODEL = "BAAI/bge-base-en-v1.5"
WEEK = 7 * 86400
YEAR = 31557600


def item_side():
    d = find("ext_items.csv")
    items = pd.read_csv(d / "ext_items.csv")
    genome = np.load(d / "ext_genome.npy").astype(np.float32)
    assert len(items) == len(genome) == 8237 and items.item.is_unique
    year = np.full(N_ITEMS, np.nan, np.float32)
    year[items.item] = items.year
    text = [f"{t} ({'' if np.isnan(y) else int(y)}). Genres: {g.replace('|', ', ')}."
            for t, y, g in zip(items.title, items.year, items.genres)]
    model = SentenceTransformer(TEXT_MODEL, device="cuda" if torch.cuda.is_available() else "cpu")
    T = model.encode(text, batch_size=128, normalize_embeddings=True, show_progress_bar=False)
    has = ~np.isnan(genome).any(1)
    gz = (genome - np.nanmean(genome, 0)) / (np.nanstd(genome, 0) + 1e-6)
    gz = np.where(has[:, None], gz, 0)
    gz /= np.linalg.norm(gz, axis=1, keepdims=True) + 1e-9
    E = np.zeros((N_ITEMS, gz.shape[1] + T.shape[1]), np.float32)
    E[items.item] = np.hstack([gz, T]) / np.sqrt(1 + has[:, None])
    print(f"content: {E.shape[1]} dims, genome for {has.mean():.3f} of items, year for {np.isfinite(year[items.item]).mean():.3f}")
    return year, E


def gram(X, Y, chunk=8192):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    G = torch.zeros((N_ITEMS, N_ITEMS), dtype=torch.float32, device=dev)
    GY = torch.zeros_like(G)
    for a in range(0, X.shape[0], chunk):
        x = torch.from_numpy(X[a:a + chunk].toarray()).to(dev)
        G += x.T @ x
        GY += x.T @ torch.from_numpy(Y[a:a + chunk].toarray()).to(dev)
    return G.cpu().numpy().astype(np.float64), GY.cpu().numpy().astype(np.float64)


def external():
    t = time.time()
    z = np.load(find("ext_ratings.npz") / "ext_ratings.npz")
    user, item, ts = z["user"].astype(np.int64), z["item"].astype(np.int64), z["timestamp"].astype(np.int64)
    rating = z["rating"] / 2
    like = rating >= LIKE
    n = user.max() + 1
    X = sp.csr_matrix((np.ones(len(user), np.float32), (user, item)), shape=(n, N_ITEMS))
    Y = sp.csr_matrix((like.astype(np.float32), (user, item)), shape=(n, N_ITEMS))
    Y.eliminate_zeros()
    G, GY = gram(X, Y)
    o = np.lexsort((ts, user))
    seq = {"user": user[o], "item": item[o], "ts": ts[o], "like": like[o]}
    t0 = ts.min()
    w = (ts - t0) // WEEK
    nw = int(w.max()) + 2

    def cum(m):
        c = np.bincount(item[m] * nw + w[m] + 1, minlength=N_ITEMS * nw).reshape(N_ITEMS, nw)
        return np.cumsum(c, 1).astype(np.float32)

    cnt = np.bincount(item, minlength=N_ITEMS).astype(np.float32)
    launch = np.full(N_ITEMS, np.inf)
    np.minimum.at(launch, item, w.astype(float))
    launch[cnt == 0] = -np.inf
    ext = {"G": G, "GY": GY, "seq": seq, "t0": t0, "launch": launch, "cum_all": cum(np.ones(len(item), bool)), "cum_like": cum(like),
           "cnt": cnt, "rating": np.where(cnt > 0, np.bincount(item, rating, N_ITEMS) / np.maximum(cnt, 1), np.nan),
           "like": np.where(cnt > 0, np.bincount(item, like, N_ITEMS) / np.maximum(cnt, 1), np.nan), "cache": {}}
    print(f"external: {len(user)} ratings, {n} users, {(cnt > 0).sum()} items, {nw} weeks ({time.time() - t:.0f}s)",
          flush=True)
    return ext


def transitions(seq, steps, session, m=None):
    u, i, t = seq["user"], seq["item"], seq["ts"]
    if m is not None:
        u, i, t = u[m], i[m], t[m]
    T = sp.csr_matrix((N_ITEMS, N_ITEMS), dtype=np.float32)
    for d in range(1, steps + 1):
        ok = u[:-d] == u[d:]
        if session:
            ok &= t[d:] - t[:-d] <= session
        T = T + sp.csr_matrix((np.full(ok.sum(), 1 / d, np.float32), (i[:-d][ok], i[d:][ok])), shape=(N_ITEMS, N_ITEMS))
    return T


def era_window(ext, y0, y1):
    ts = ext["seq"]["ts"]
    return (ts >= (y0 - 1970) * YEAR) & (ts < (y1 - 1970) * YEAR)


def user_week(df, n_users, t0):
    last = df.groupby("userId").timestamp.max().reindex(range(n_users))
    w = ((last - t0) // WEEK).to_numpy()
    return np.where(np.isnan(w), np.nanmedian(w), w).astype(np.int64)
