import collections
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

N_CLASSES = 6
N_NODE_LABELS = 3
N_NODE_FEATS = 18
WL_ITERS = 4
PROP_HOPS = 3
WWL_HOPS = (1, 2, 3, 4, 5)
FGW_ALPHAS = (0.5, 0.75)
RWSE_STEPS = 16
WORK = Path("/tmp/work")
OUT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")


def log(msg):
    print(msg, flush=True)
    WORK.mkdir(parents=True, exist_ok=True)
    with open(WORK / "log.txt", "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def find_data():
    roots = [Path("/kaggle/input")]
    try:
        roots.append(Path(__file__).resolve().parent.parent / "data")
    except NameError:
        roots.append(Path.cwd() / "data")
    for root in roots:
        if not root.exists():
            continue
        for p in sorted(root.rglob("train.csv")):
            with open(p, encoding="utf-8") as f:
                if "edge_index" in f.readline() and (p.parent / "test.csv").exists():
                    return p.parent
    raise FileNotFoundError("no train.csv with an edge_index column")


def parse(df):
    graphs = []
    for r in df.itertuples(index=False):
        x = np.asarray(json.loads(r.node_features), dtype=np.float64).reshape(-1, N_NODE_FEATS)
        nl = np.asarray(json.loads(r.node_labels), dtype=np.int64) - 1
        e = np.asarray(json.loads(r.edge_index), dtype=np.int64).reshape(2, -1)
        n = int(r.num_nodes)
        assert len(x) == len(nl) == n, r.graph_id
        assert nl.min() >= 0 and nl.max() < N_NODE_LABELS, r.graph_id
        assert e.size == 0 or (e.min() >= 0 and e.max() < n), r.graph_id
        graphs.append({"id": r.graph_id, "x": x, "nl": nl, "e_raw": e})
    return graphs


def undirected(e):
    e = e[:, e[0] != e[1]]
    if e.shape[1] == 0:
        return e
    return np.unique(np.concatenate([e, e[::-1]], axis=1), axis=1)


def signature(g):
    e = np.sort(undirected(g["e_raw"]), axis=0) if g["e_raw"].size else g["e_raw"]
    return hashlib.md5(g["x"].tobytes() + g["nl"].tobytes() + e.tobytes()).hexdigest()


def data_checks(graphs, n_train):
    sym = self_loops = 0
    for g in graphs:
        pairs = set(map(tuple, g["e_raw"].T))
        sym += all((b, a) in pairs for a, b in pairs)
        self_loops += sum(a == b for a, b in pairs)
    log(f"graphs with symmetric edge lists: {sym}/{len(graphs)}; self-loops: {self_loops}")
    sig = [signature(g) for g in graphs]
    train_sig = collections.Counter(sig[:n_train])
    log(f"exact duplicates: within train {sum(v - 1 for v in train_sig.values())}, "
        f"test graphs also in train {sum(s in train_sig for s in sig[n_train:])}")


def slog(x):
    return np.sign(x) * np.log1p(np.abs(x))


def prepare(graphs, n_train):
    nodes = np.vstack([slog(g["x"]) for g in graphs[:n_train]])
    mu, sd = nodes.mean(0), nodes.std(0) + 1e-6
    for g in graphs:
        g["e"] = undirected(g["e_raw"])
        g["h"] = np.hstack([(slog(g["x"]) - mu) / sd, np.eye(N_NODE_LABELS)[g["nl"]]]).astype(np.float32)


def propagate(h, e):
    n = len(h)
    deg = np.bincount(e[0], minlength=n)
    agg = np.zeros_like(h)
    np.add.at(agg, e[0], h[e[1]])
    return np.where(deg[:, None] > 0, agg / np.maximum(deg, 1)[:, None], h)


def hop_embeddings(g):
    h = g["h"].astype(np.float64)
    hops = [h]
    for _ in range(max(WWL_HOPS)):
        h = 0.5 * (h + propagate(h, g["e"]))
        hops.append(h)
    return hops


def rwse(g):
    n = len(g["h"])
    A = np.zeros((n, n))
    A[g["e"][0], g["e"][1]] = 1.0
    deg = A.sum(1, keepdims=True)
    P = np.divide(A, deg, out=np.zeros_like(A), where=deg > 0)
    out, M = [], np.eye(n)
    for _ in range(RWSE_STEPS):
        M = M @ P
        out.append(np.diag(M))
    return np.stack(out, 1)


def pooled_features(g):
    e, n = g["e"], len(g["h"])
    parts = []
    for h in g["hops"][:PROP_HOPS + 1]:
        parts += [h.mean(0), h.std(0), h.max(0), h.min(0)]
    deg = np.bincount(e[0], minlength=n)
    m = e.shape[1] / 2
    struct = [n, np.log(n), m, 2 * m / (n * (n - 1)) if n > 1 else 0.0, deg.mean(), deg.std(), deg.max(),
              (deg == 0).mean()]
    struct += list(np.bincount(g["nl"], minlength=N_NODE_LABELS) / n)
    return np.concatenate(parts + [np.asarray(struct)])


def wl_kernel(graphs):
    labs = [g["nl"].copy() for g in graphs]
    nbrs = []
    for g in graphs:
        nb = [[] for _ in range(len(g["nl"]))]
        for a, b in g["e"].T:
            nb[a].append(b)
        nbrs.append(nb)
    counts = [collections.Counter() for _ in graphs]
    for it in range(WL_ITERS + 1):
        for c, lab in zip(counts, labs):
            c.update((it, int(v)) for v in lab)
        if it == WL_ITERS:
            break
        table = {}
        labs = [np.array([table.setdefault((int(lab[i]), tuple(sorted(int(lab[j]) for j in nb[i]))), len(table))
                          for i in range(len(lab))], dtype=np.int64)
                for lab, nb in zip(labs, nbrs)]
    keys = {k: i for i, k in enumerate(sorted(set().union(*counts)))}
    M = np.zeros((len(graphs), len(keys)), dtype=np.float64)
    for r, c in enumerate(counts):
        for k, v in c.items():
            M[r, keys[k]] = v
    K = M @ M.T
    d = np.sqrt(np.diag(K))
    log(f"WL patterns: {len(keys)}")
    return K / np.outer(d, d)


def import_ot():
    try:
        import ot
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pot"], check=True)
        import ot
    return ot


def pairwise(row, n):
    from joblib import Parallel, delayed
    rows = Parallel(n_jobs=-1)(delayed(row)(i) for i in range(n))
    out = {}
    for key in rows[0][0]:
        D = np.zeros((n, n))
        for i, r in enumerate(rows):
            D[i, i + 1:] = [d[key] for d in r]
        D = D + D.T
        assert np.isfinite(D).all() and (D >= -1e-9).all(), key
        out[key] = np.clip(D, 0, None)
    return out


def wasserstein_distances(graphs):
    ot = import_ot()
    emb = [np.hstack(g["hops"]) for g in graphs]
    d = graphs[0]["h"].shape[1]
    w = [np.full(len(x), 1.0 / len(x)) for x in emb]

    def row(i):
        out = []
        for j in range(i + 1, len(emb)):
            dists = {}
            for k in WWL_HOPS:
                a, b = emb[i][:, :(k + 1) * d], emb[j][:, :(k + 1) * d]
                dists[k] = ot.emd2(w[i], w[j], ot.dist(a, b, metric="euclidean"))
            out.append(dists)
        return out

    return pairwise(row, len(emb))


def shortest_paths(g):
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path
    n = len(g["h"])
    A = csr_matrix((np.ones(g["e"].shape[1]), (g["e"][0], g["e"][1])), shape=(n, n))
    S = shortest_path(A, directed=False, unweighted=True)
    fin = np.isfinite(S)
    S[~fin] = S[fin].max() + 1
    return S / S.max() if S.max() > 0 else S


def structural_encodings(graphs, n_train):
    enc = [np.hstack([g["rw"], np.log1p(np.bincount(g["e"][0], minlength=len(g["h"])))[:, None]]) for g in graphs]
    nodes = np.vstack(enc[:n_train])
    mu, sd = nodes.mean(0), nodes.std(0) + 1e-6
    return [(x - mu) / sd for x in enc]


def fgw_distances(graphs, n_train):
    ot = import_ot()
    plain = [np.hstack(g["hops"][:PROP_HOPS + 1]) for g in graphs]
    views = {"plain": plain, "struct": [np.hstack([p, s]) for p, s in zip(plain, structural_encodings(graphs, n_train))]}
    sp = [shortest_paths(g) for g in graphs]
    w = [np.full(len(x), 1.0 / len(x)) for x in plain]
    pick = np.random.default_rng(0).choice(sum(map(len, plain)), 2000, replace=False)
    scale = {}
    for v, emb in views.items():
        sample = np.vstack(emb)[pick]
        scale[v] = np.median(ot.dist(sample, sample, metric="euclidean"))
        log(f"FGW {v} feature cost scale (median node distance): {scale[v]:.3f}")

    def row(i):
        out = []
        for j in range(i + 1, len(plain)):
            d = {}
            for v, emb in views.items():
                M = ot.dist(emb[i], emb[j], metric="euclidean") / scale[v]
                for a in FGW_ALPHAS:
                    d[f"{v}_{a}"] = float(ot.gromov.fused_gromov_wasserstein2(M, sp[i], sp[j], w[i], w[j],
                                                                             loss_fun="square_loss", alpha=a))
            out.append(d)
        return out

    return pairwise(row, len(plain))


def load():
    d = find_data()
    tr, te = pd.read_csv(d / "train.csv"), pd.read_csv(d / "test.csv")
    sub = pd.read_csv(d / "sample_submission.csv")
    assert list(sub.graph_id) == list(te.graph_id)
    graphs = parse(tr) + parse(te)
    y = tr.label.to_numpy() - 1
    assert sorted(set(y)) == list(range(N_CLASSES))
    log(f"data {d}: train {len(tr)}, test {len(te)}, class counts {np.bincount(y).tolist()}")
    data_checks(graphs, len(tr))
    prepare(graphs, len(tr))
    for g in graphs:
        g["hops"] = hop_embeddings(g)
        g["rw"] = rwse(g)
        g["hg"] = np.hstack([g["h"], g["rw"]]).astype(np.float32)
    F = np.vstack([pooled_features(g) for g in graphs])
    assert np.isfinite(F).all()
    K = wl_kernel(graphs)
    D_wwl = wasserstein_distances(graphs)
    log(f"pooled features {F.shape}, WL kernel {K.shape}, Wasserstein distances for hops {list(D_wwl)}")
    D_fgw = fgw_distances(graphs, len(tr))
    log(f"FGW distances {list(D_fgw)}")
    return graphs, y, F, K, D_wwl, D_fgw, sub
