import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

DATA = Path(__file__).resolve().parent.parent / "data"
SMALL = DATA / "external" / "movielens" / "ml-latest-small"
GENOME = ML25 = DATA / "external" / "ml-25m"
OUT = DATA / "recommender-ext.zip"
OVERLAP = 0.7


def align():
    tr = pd.read_csv(DATA / "train.csv").sort_values(["userId", "time_idx"])
    raw = pd.read_csv(SMALL / "ratings.csv").sort_values(["userId", "timestamp"], kind="stable")
    seqs = {u: g for u, g in raw.groupby("userId")}
    votes = defaultdict(Counter)
    matched, times = 0, []
    for u, g in tr.groupby("userId"):
        r = g.rating.to_numpy()
        hits = [v for v, q in seqs.items() if len(q) >= len(r) and np.array_equal(q.rating.to_numpy()[:len(r)], r)]
        if len(hits) != 1:
            continue
        prefix = seqs[hits[0]].iloc[:len(r)]
        matched += 1
        times.append(pd.DataFrame({"userId": u, "time_idx": g.time_idx.to_numpy(), "movieId": prefix.movieId.to_numpy(),
                                   "timestamp": prefix.timestamp.to_numpy()}))
        for a, b in zip(g.movieId, prefix.movieId):
            votes[a][b] += 1
    m = {a: c.most_common(1)[0][0] for a, c in votes.items()}
    purity = np.mean([c.most_common(1)[0][1] / sum(c.values()) for c in votes.values()])
    assert purity == 1.0 and len(set(m.values())) == len(m) == tr.movieId.nunique(), (purity, len(m))
    print(f"matched {matched} users, mapped {len(m)} items, purity {purity}")
    times = pd.concat(times, ignore_index=True)
    inv = {b: a for a, b in m.items()}
    check = tr.merge(times, on=["userId", "time_idx"], validate="one_to_one")
    assert (check.movieId_x == check.movieId_y.map(inv)).all()
    items = pd.DataFrame({"item": list(m), "movieId": list(m.values())}).sort_values("item").reset_index(drop=True)
    return items, times


def rating_overlap(r, items):
    tr = pd.read_csv(DATA / "train.csv")
    mid = tr.movieId.map(dict(zip(items.item, items.movieId))).to_numpy()
    keys = np.r_[mid.astype(np.int64) * 10 + (tr.rating.to_numpy() * 2).astype(np.int64),
                 r.movieId.to_numpy().astype(np.int64) * 10 + (r.rating.to_numpy() * 2).astype(np.int64)]
    cols = pd.factorize(keys)[0]
    nc, nt = cols.max() + 1, len(tr)
    A = sp.csr_matrix((np.ones(nt, np.float32), (tr.userId, cols[:nt])), shape=(tr.userId.max() + 1, nc))
    ml, ml_idx = np.unique(r.userId.to_numpy(), return_inverse=True)
    B = sp.csr_matrix((np.ones(len(r), np.float32), (ml_idx, cols[nt:])), shape=(len(ml), nc))
    O = (A @ B.T).tocsr()
    share = O.max(1).toarray().ravel() / np.asarray(A.sum(1)).ravel()
    best = ml[np.asarray(O.argmax(1)).ravel()]
    return pd.Series(best[share >= OVERLAP]).drop_duplicates()


def external_ratings(items, times):
    r = pd.read_csv(ML25 / "ratings.csv", dtype={"userId": np.int32, "movieId": np.int32, "rating": np.float32,
                                                "timestamp": np.int64})
    print(f"ML-25M: {len(r)} ratings, {r.userId.nunique()} users, timestamps "
          f"{pd.to_datetime(r.timestamp.min(), unit='s').date()} .. {pd.to_datetime(r.timestamp.max(), unit='s').date()}")
    ours = pd.MultiIndex.from_frame(times[["movieId", "timestamp"]])
    hit = pd.MultiIndex.from_frame(r[["movieId", "timestamp"]]).isin(ours)
    k = pd.Series(hit).groupby(r.userId.to_numpy()).sum()
    drop = k[k > 0]
    best = r[hit].merge(times[["movieId", "timestamp", "userId"]].rename(columns={"userId": "comp"}),
                        on=["movieId", "timestamp"]).groupby("comp").userId.agg(lambda x: x.value_counts().iloc[0])
    n_ours = times.groupby("userId").size()
    cover = (best / n_ours.reindex(best.index))
    print(f"dropping {len(drop)} ML-25M users sharing a (movie, timestamp) with our prefixes; "
          f"shared counts: {drop.describe()[['min', '50%', 'max']].to_dict()}")
    print(f"our users found in ML-25M: {len(best)} of {len(n_ours)}; share of each prefix found in its best match: "
          f"min {cover.min():.3f}, median {cover.median():.3f}")
    extra = rating_overlap(r, items)
    extra = extra[~extra.isin(drop.index)]
    print(f"also dropping {len(extra)} ML-25M users whose (movie, rating) pairs cover >= {OVERLAP:.0%} of one of our "
          f"users' train (timestamp-free rule; catches user 475 and any re-timestamped account): {sorted(extra.tolist())}")
    r = r[~r.userId.isin(drop.index) & ~r.userId.isin(extra)]
    r = r[r.movieId.isin(items.movieId)]
    r["item"] = r.movieId.map(dict(zip(items.movieId, items.item))).astype(np.int16)
    r["user"] = pd.factorize(r.userId)[0].astype(np.int32)
    print(f"external kept: {len(r)} ratings, {r.user.nunique()} users, {r.item.nunique()} of {len(items)} items")
    return r


def main():
    items, times = align()
    mv = pd.read_csv(SMALL / "movies.csv")
    items = items.merge(mv, on="movieId", how="left", validate="one_to_one")
    items["year"] = items.title.str.extract(r"\((\d{4})\)\s*$", expand=False).astype(float)
    items["title"] = items.title.str.replace(r"\s*\(\d{4}\)\s*$", "", regex=True)
    items["title"] = items.title.map(lambda t: re.sub(r"^(.*), (The|A|An)$", r"\2 \1", t))
    print(f"year known for {items.year.notna().mean():.3f}, title sample: {items.title.head(3).tolist()}")

    gs = pd.read_csv(GENOME / "genome-scores.csv")
    gs = gs[gs.movieId.isin(items.movieId)]
    G = gs.pivot(index="movieId", columns="tagId", values="relevance").reindex(items.movieId)
    print(f"genome: {G.shape[1]} tags, covers {G.notna().all(1).mean():.3f} of items")
    tmp = DATA / "ext_genome.npy"
    np.save(tmp, G.to_numpy(np.float16))
    items[["item", "movieId", "title", "genres", "year"]].to_csv(DATA / "ext_items.csv", index=False)
    times[["userId", "time_idx", "timestamp"]].to_csv(DATA / "ext_times.csv", index=False)
    r = external_ratings(items, times)
    np.savez(DATA / "ext_ratings.npz", user=r.user.to_numpy(), item=r.item.to_numpy(),
             rating=(r.rating.to_numpy() * 2).astype(np.int8), timestamp=r.timestamp.to_numpy().astype(np.int32))
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for f in ("ext_items.csv", "ext_genome.npy", "ext_times.csv", "ext_ratings.npz"):
            z.write(DATA / f, f)
            (DATA / f).unlink()
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
