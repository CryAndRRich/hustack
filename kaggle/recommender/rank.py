import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from content import WEEK
from models import active
from preprocess import LIKE, N_ITEMS, TRAIN_SHARE

K = 20
C = 300
PARAMS = {"objective": "lambdarank", "metric": "ndcg", "eval_at": [K], "learning_rate": 0.03, "num_leaves": 31,
          "min_child_samples": 50, "colsample_bytree": 0.8, "subsample": 0.8, "subsample_freq": 1, "verbose": -1}


SESSION = 3600


def mask(scores, X, allowed):
    s = np.array(scores, np.float32)
    s[X.nonzero()] = -np.inf
    s[:, ~active(X)] = -np.inf
    s[~allowed] = -np.inf
    return s


def recommend(scores, X, allowed):
    s = mask(scores, X, allowed)
    top = np.argpartition(-s, K, axis=1)[:, :K]
    return np.take_along_axis(top, np.argsort(-np.take_along_axis(s, top, 1), 1), 1)


def metrics(top, targets):
    disc = 1 / np.log2(np.arange(2, K + 2))
    m = {"recall": [], "recall_cap": [], "ndcg": [], "map": []}
    for u, t in targets.items():
        hit = np.isin(top[u], t)
        n = min(len(t), K)
        m["recall"].append(hit.sum() / len(t))
        m["recall_cap"].append(hit.sum() / n)
        m["ndcg"].append((hit * disc).sum() / disc[:n].sum())
        m["map"].append((np.cumsum(hit) / np.arange(1, K + 1) * hit).sum() / n)
    return {k: float(np.mean(v)) for k, v in m.items()}


def candidates(scores, X, allowed, users):
    tops = np.concatenate([np.argpartition(-mask(s, X, allowed)[users], C, axis=1)[:, :C] for s in scores.values()], 1)
    cand = [np.unique(row) for row in tops]
    return np.repeat(users, [len(c) for c in cand]), np.concatenate(cand)


def labels(u, i, targets):
    sets = {k: set(v) for k, v in targets.items()}
    return np.array([it in sets[us] for us, it in zip(u, i)], np.float32)


def features(df, X, scores, side, u, i):
    S, E, year = side["S"], side["E"], side["year"]
    n = df.groupby("userId").time_idx.transform("size")
    d = df.assign(pos=df.time_idx / (n - 1).clip(lower=1), late=df.time_idx >= np.round(n * TRAIN_SHARE),
                  like=df.rating >= LIKE, res=df.rating - df.groupby("userId").rating.transform("mean"))
    it = d.groupby("movieId").agg(i_cnt=("userId", "size"), i_rating=("rating", "mean"), i_like=("like", "mean"),
                                  i_pos=("pos", "mean"), i_late=("late", "mean"), i_res=("res", "sum")).reindex(range(N_ITEMS))
    it["i_bias"] = it.i_res / (it.i_cnt + 10)
    it["i_cnt"] = np.log1p(it.i_cnt.fillna(0) / len(d) * 1e5)
    it["i_year"] = year
    us = d.groupby("userId").agg(u_n=("rating", "size"), u_rating=("rating", "mean"), u_like=("like", "mean"),
                                 u_std=("rating", "std")).reindex(range(X.shape[0]))
    us["u_n"] = np.log(us.u_n)
    us = us.join(session_stats(df).reindex(range(X.shape[0])))
    f = pd.concat([it.drop(columns="i_res").iloc[i].reset_index(drop=True), us.iloc[u].reset_index(drop=True)], axis=1)
    f["rating_gap"] = f.i_rating - f.u_rating
    for name, s in scores.items():
        order = np.argsort(-mask(s, X, side["allowed"]), axis=1)
        rank = np.empty(s.shape, np.int32)
        np.put_along_axis(rank, order, np.arange(N_ITEMS, dtype=np.int32)[None, :], 1)
        f[f"{name}_score"], f[f"{name}_rank"] = s[u, i], rank[u, i]
    hist = {user: (g.movieId.to_numpy()[::-1], g.res.to_numpy()[::-1]) for user, g in d.groupby("userId")}
    cols = ["sim_last1", "sim_last5", "sim_last20", "csim_last1", "csim_last5", "csim_last20",
            "rating_cf", "rating_content", "year_gap", "u_year_std"]
    out = np.zeros((len(u), len(cols)), np.float32)
    starts = np.r_[0, np.flatnonzero(np.diff(u)) + 1, len(u)]
    for a, b in zip(starts[:-1], starts[1:]):
        h, res = hist[u[a]]
        c = i[a:b]
        s, cs = S[np.ix_(h, c)], np.maximum(E[h] @ E[c].T, 0)
        yr = year[h[:20]]
        out[a:b] = np.column_stack([
            s[0], s[:5].mean(0), s[:20].mean(0), cs[0], cs[:5].mean(0), cs[:20].mean(0),
            res @ s / (s.sum(0) + 1e-3), res @ cs / (cs.sum(0) + 1e-3),
            year[c] - np.nanmean(yr), np.full(len(c), np.nanstd(yr))])
    f[cols] = out
    f["abs_year_gap"] = f.year_gap.abs()
    ext, wu = side["ext"], side["week"][u]
    f["ext_cnt"], f["ext_rating"], f["ext_like"] = np.log1p(ext["cnt"][i]), ext["rating"][i], ext["like"][i]
    last = ext["cum_all"].shape[1] - 1
    for kind in ("like", "all"):
        C = ext[f"cum_{kind}"]
        tot = C.sum(0)
        for lo, hi in ((0, 4), (0, 13), (0, 52), (-13, 0)):
            a, b = np.clip(wu + lo, 0, last), np.clip(wu + hi, 0, last)
            f[f"tp_{kind}_{lo}_{hi}"] = (C[i, b] - C[i, a]) / np.maximum(tot[b] - tot[a], 1) * 1e4
    f["tp_trend"] = np.log1p(f.tp_all_0_13) - np.log1p(f["tp_all_-13_0"])
    f["u_time"] = 1970 + (ext["t0"] + wu * WEEK) / 31557600
    f["item_age"] = f.u_time - f.i_year
    launch = ext["launch"][i]
    f["weeks_since_launch"] = np.where(np.isfinite(launch), wu - launch, np.nan)
    return f.astype(np.float32)


def session_stats(df):
    rows = {}
    for user, ts in df.groupby("userId").timestamp:
        t = ts.to_numpy()
        if np.isnan(t).any():
            continue
        gap = np.diff(t)
        brk = np.flatnonzero(gap > SESSION)
        start = brk[-1] + 1 if len(brk) else 0
        rows[user] = (len(t) - start, (t[-1] - t[start]) / 60, gap[-1] / 3600 if len(gap) else np.nan,
                      (t[-1] - t[0]) / 86400, len(brk) + 1)
    return pd.DataFrame.from_dict(rows, orient="index",
                                  columns=["sess_n", "sess_minutes", "last_gap_hours", "span_days", "n_sessions"])


def groups(u):
    return np.unique(u, return_counts=True)[1]


def fit(F, y, g, rounds, valid=None, seed=0):
    ds = lgb.Dataset(F, y, group=groups(g))
    if valid is None:
        return lgb.train({**PARAMS, "seed": seed}, ds, rounds)
    vs = lgb.Dataset(valid[0], valid[1], group=groups(valid[2]), reference=ds)
    return lgb.train(PARAMS, ds, rounds, valid_sets=[vs], callbacks=[lgb.early_stopping(100, verbose=False)])


def cross_validate(F, y, g, u, rounds, folds=5):
    oof, iters, users = np.zeros(len(y)), [], np.unique(u)
    for a, _ in KFold(folds, shuffle=True, random_state=0).split(users):
        tr = np.isin(u, users[a])
        model = fit(F[tr], y[tr], g[tr], rounds, (F[~tr], y[~tr], g[~tr]))
        oof[~tr] = model.predict(F[~tr], num_iteration=model.best_iteration)
        iters.append(model.best_iteration)
    return oof, max(1, int(np.mean(iters)))


def top_from_pairs(u, i, p, n_users):
    top = np.zeros((n_users, K), np.int64)
    o = np.lexsort((-p, u))
    u, i = u[o], i[o]
    for s in np.r_[0, np.flatnonzero(np.diff(u)) + 1]:
        top[u[s]] = i[s:s + K]
    return top
