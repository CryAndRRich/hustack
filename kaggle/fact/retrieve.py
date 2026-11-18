import heapq

import numpy as np
import pandas as pd
import scipy.sparse as sp
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import HashingVectorizer

from preprocess import K_CANDS, bigrams, clean, log, match_key, parse_lines, title_base

HASH = HashingVectorizer(analyzer="char", ngram_range=(2, 2), n_features=2 ** 23, alternate_sign=False,
                         norm=None, dtype=np.float32)
TOP_BM25 = 20
POOL_PAGES = 25
LINK_FROM = 3
MAX_LINKS = 30
MAX_VARIANTS = 5
MAX_LINES = 300
DF_MAX_FRAC = 0.05
HOP_FROM = 3
HOP_PAGES = 6
HOP_SENTS = 8


def _vectorize(titles, texts):
    docs = [(clean(t.replace("_", " ")) + " ") * 3 + clean(x[:1500]) for t, x in zip(titles, texts)]
    return HASH.transform(docs)


def build_bm25(titles, texts, k1=1.2, b=0.75):
    n = len(titles)
    step = 50000
    parts = Parallel(n_jobs=4)(delayed(_vectorize)(titles[a:a + step], texts[a:a + step]) for a in range(0, n, step))
    X = sp.vstack(parts).tocsr()
    del parts
    df = np.bincount(X.indices, minlength=X.shape[1])
    idf = np.log1p((n - df + 0.5) / (df + 0.5)).astype(np.float32)
    stop = df > DF_MAX_FRAC * n
    del df
    dl = np.asarray(X.sum(axis=1), dtype=np.float32).ravel()
    norm = (k1 * (1 - b + b * dl / dl.mean())).astype(np.float32)
    denom = norm[np.repeat(np.arange(n, dtype=np.int32), np.diff(X.indptr))]
    denom += X.data
    X.data *= k1 + 1
    X.data /= denom
    del denom
    X.data *= idf[X.indices]
    X.data[stop[X.indices]] = 0
    X.eliminate_zeros()
    log(f"bm25: {n} pages, nnz {X.nnz / 1e6:.0f}M")
    return X.T.tocsr()


class Retriever:
    def __init__(self, titles, texts, lines):
        self.titles, self.lines = titles, lines
        self.title2idx = {t: i for i, t in enumerate(titles)}
        self.base2idx = {}
        for i, t in enumerate(titles):
            bkey = title_base(t)
            if len(bkey) >= 2:
                self.base2idx.setdefault(bkey, []).append(i)
        self.max_base = max(map(len, self.base2idx))
        self.WT = build_bm25(titles, texts)
        self.cache = {}

    def page(self, i):
        p = self.cache.get(i)
        if p is None:
            p = parse_lines(self.lines[i])[:MAX_LINES]
            if len(self.cache) > 30000:
                self.cache.clear()
            self.cache[i] = p
        return p

    def resolve(self, target):
        i = self.title2idx.get(target.replace(" ", "_"))
        return self.title2idx.get(target) if i is None else i

    def title_matches(self, claim):
        c = match_key(claim)
        spans = []
        for a in range(len(c)):
            for e in range(min(len(c), a + self.max_base), a + 1, -1):
                if c[a:e] in self.base2idx:
                    spans.append((a, e))
        out = {}
        for a, e in spans:
            maximal = not any(a2 <= a and e <= e2 and (a2, e2) != (a, e) for a2, e2 in spans)
            bonus = 1.0 if maximal else 0.4
            for i in self.base2idx[c[a:e]]:
                out[i] = max(out.get(i, 0.0), bonus * (0.8 if self.titles[i].endswith(")") else 1.0))
        return out

    def candidates(self, claim, s_idx, s_val):
        top = s_idx[np.argpartition(-s_val, TOP_BM25)[:TOP_BM25]] if len(s_val) > TOP_BM25 else s_idx
        rmax = float(s_val.max()) if len(s_val) else 1.0

        def bm(ids):
            ids = np.asarray(ids, dtype=np.int64)
            if not len(s_idx):
                return np.zeros(len(ids))
            pos = np.minimum(np.searchsorted(s_idx, ids), len(s_idx) - 1)
            return np.where(s_idx[pos] == ids, s_val[pos], 0.0) / rmax

        tm = self.title_matches(claim)
        by_base = {}
        for i in tm:
            by_base.setdefault(title_base(self.titles[i]), []).append(i)
        keep = []
        for ids in by_base.values():
            if len(ids) > MAX_VARIANTS:
                ids = [ids[j] for j in np.argsort(-bm(ids))[:MAX_VARIANTS]]
            keep.extend(ids)
        ids = list(dict.fromkeys(keep + top.tolist()))
        score = bm(ids) + np.array([tm.get(i, 0.0) for i in ids])
        order = np.argsort(-score)[:POOL_PAGES]
        pool = {ids[j]: float(score[j]) for j in order}
        links = {}
        for i in list(pool)[:LINK_FROM]:
            for _, _, targets in self.page(i):
                for t in targets:
                    j = self.resolve(t)
                    if j is not None and j not in pool:
                        links[j] = None
                if len(links) >= MAX_LINKS:
                    break
            if len(links) >= MAX_LINKS:
                break
        links = list(links)[:MAX_LINKS]
        for j, s in zip(links, bm(links)):
            pool[j] = float(s)
        cb = list(bigrams(clean(claim)))
        denom = max(len(cb), 1)
        sents = []
        for i, ps in pool.items():
            for sid, sent, _ in self.page(i):
                sents.append((ps + sum(map(sent.__contains__, cb)) / denom + (0.15 if sid == 0 else 0.0), i, sid, sent))
        return list(pool), heapq.nlargest(K_CANDS, sents, key=lambda x: x[0])

    def run(self, claims):
        q = HASH.transform([clean(c) for c in claims["claim"]])
        q.data[:] = 1.0
        rows, pools = [], []
        qids = claims.index.to_numpy()
        for a in range(0, len(claims), 64):
            S = (q[a:a + 64] @ self.WT).tocsr()
            S.sort_indices()
            for r in range(S.shape[0]):
                lo, hi = S.indptr[r], S.indptr[r + 1]
                qid = qids[a + r]
                pool, sents = self.candidates(claims.at[qid, "claim"], S.indices[lo:hi].astype(np.int64), S.data[lo:hi])
                pools.append(pool)
                for rank, (pre, i, sid, sent) in enumerate(sents):
                    rows.append((qid, self.titles[i], sid, sent, pre, rank))
            if a % 6400 == 0:
                log(f"retrieval {a}/{len(claims)}")
        cands = pd.DataFrame(rows, columns=["qid", "title", "sid", "text", "pre", "rank"])
        return cands, pools

    def hop_candidates(self, cands):
        rows = []
        for qid, d in cands.groupby("qid", sort=False):
            have = set(zip(d["title"], d["sid"]))
            pages = {}
            for t, s in zip(d["title"].iloc[:HOP_FROM], d["sid"].iloc[:HOP_FROM]):
                targets = next((tg for sid, _, tg in self.page(self.title2idx[t]) if sid == s), [])
                for tg in targets:
                    j = self.resolve(tg)
                    if j is not None and self.titles[j] != t:
                        pages.setdefault(j, None)
            for j in list(pages)[:HOP_PAGES]:
                for sid, sent, _ in self.page(j)[:HOP_SENTS]:
                    if (self.titles[j], sid) not in have:
                        rows.append((qid, self.titles[j], sid, sent))
        return pd.DataFrame(rows, columns=["qid", "title", "sid", "text"])

    def gold_texts(self, claims):
        rows = []
        for qid, groups in claims["groups"].items():
            for title, sid in {k for g in groups for k in g}:
                text = next((s for sd, s, _ in self.page(self.title2idx[title]) if sd == sid), "")
                rows.append((qid, title, sid, text))
        return pd.DataFrame(rows, columns=["qid", "title", "sid", "text"])


def retrieval_report(claims, cands, pools, titles):
    lab = claims[claims["groups"].map(len) > 0]
    if lab.empty:
        return
    keys = {q: set(zip(d["title"], d["sid"])) for q, d in cands.groupby("qid")}
    pool_titles = {qid: {titles[i] for i in p} for qid, p in zip(claims.index, pools)}
    for split, d in lab.groupby("split"):
        page_hit = np.mean([any(all(t in pool_titles[q] for t, _ in g) for g in gs) for q, gs in d["groups"].items()])
        sent_hit = np.mean([any(all(k in keys.get(q, set()) for k in g) for g in gs) for q, gs in d["groups"].items()])
        log(f"retrieval {split}: S/R claims {len(d)}  gold group pages in pool {page_hit:.4f}  "
            f"gold group sentences in top-{K_CANDS} {sent_hit:.4f}")
