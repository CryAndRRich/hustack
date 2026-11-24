import multiprocessing as mp
import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
import torch.nn as nn

from benchmark_utils.base_solver import GeneCompetSolver

ARM = "v7.4"
ARMS = ("v7.0", "v7.1", "v7.2", "v7.3", "v7.4", "v7.5")
STAGE = ARMS.index(ARM)
PSEUDO_A_OPERON, USE_OPERON, USE_XKNN, USE_SHIFT = STAGE >= 1, STAGE >= 2, STAGE >= 3, STAGE >= 4
GENE_MODEL = "prom2" if STAGE >= 5 else "prom"
SEEDS = (0, 1, 2, 3, 4)
SPLIT_SEED = 1234
RANK = 32
STEPS, BATCH, WARMUP = 500, 1_000_000, 25
EVAL_EVERY = 50
ABSORB_STEPS = 200
FIT_BUDGET, STACK_RESERVE, MIN_SEED_SECONDS = 2900.0, 300.0, 150.0
LR_FREE, LR_NET, WD_NET, L2_FREE = 1e-2, 2e-3, 1e-4, 1e-4
P_DROP = 0.15
HIDDEN, EMB_GENOME, EMB_CAT, EMB_WORD = 256, 8, 4, 16
OPERON_GAP = 100
LOG_R_MIN, LOG_R_MAX, ETA_MAX, NLL_CAP, LEVEL = -6.0, 7.0, 30.0, 100.0, 6.0
PSEUDO_C_UNITS, PSEUDO_B_PROJECTS, PSEUDO_B_GROUPS, PSEUDO_B_MAX_RUNS = 0.02, 0.02, 0.02, 50
PSEUDO_A_UNITS, PSEUDO_A_RUNS, PSEUDO_A_MIN_RUNS = 0.10, 0.17, 10
PSEUDO_A_GENE_SHARE, PSEUDO_A_OPERON_GAP = 0.30, 30
CALIB_CELLS, CHECK_CELLS, MIN_CALIB = 1_000_000, 200_000, 500
WINDOW, KNN_K, KNN_G = 3, 20, 20
STACK_MIN, STACK_ITERS = 5_000, 400
SHIFT_MIN, SHIFT_SAMPLE, SHIFT_ITERS, SHIFT_CLIP = 1_000, 200_000, 300, (0.05, 20.0)
MIN_WORD_GENES = 3
CHUNK = 1_000_000
RUN_GROUPS = ("prj_accession", "perturbation_compound", "medium_name", "growth_phase")
RUN_CATS = ("medium_category", "oxygen_condition", "perturbation_category", "is_wild_type", "condition_confidence",
            "growth_mode")
RUN_NUMS = ("temperature_celsius", "ph", "od600", "agitation_rpm", "log_size_factor")
SRA_CATS = ("instrument_platform", "instrument_model", "library_layout", "library_selection", "library_source")
SRA_NUMS = ("log_read_length", "year", "kw_stranded", "kw_rrna_depleted", "kw_total_rna", "has_protocol")
CONDITION = ("medium_name", "medium_category", "growth_phase", "perturbation_compound", "perturbation_category",
             "temperature_celsius", "ph", "od600", "oxygen_condition", "is_wild_type", "agitation_rpm", "growth_mode")
KINDS = ("A", "B", "C", "D")
THETA0 = (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0)
FREE = {"A": (0, 1, 2, 3, 4, 5, 6, 7, 8, 9), "B": (0, 1, 4, 5, 6, 7, 8), "C": (0, 1, 2, 4, 5, 7, 8), "D": (0, 1, 4, 5)}
BASIC = (0, 1, 4, 5)
DESCRIPTORS = ("lvg", "lnr", "win_n", "knn_w", "gknn_w")
DESC_NAMES = DESCRIPTORS + ("g_cells", "g_hom", "g_unit", "g_genome_genes", "g_genome_runs", "r_cells", "r_prj_known",
                            "r_lsf", "r_genome_runs")
T0 = time.time()


def log(*args):
    print(f"[{ARM} {time.time() - T0:6.0f}s]", *args, flush=True)


def pair_index(df, a, b):
    ca, cb = (df[c] if isinstance(df[c].dtype, pd.CategoricalDtype) else df[c].astype("category") for c in (a, b))
    nb = len(cb.cat.categories) + 1
    codes, uniq = pd.factorize(ca.cat.codes.to_numpy().astype(np.int64) * nb + cb.cat.codes.to_numpy())
    cats_a, cats_b = np.asarray(ca.cat.categories.astype(str)), np.asarray(cb.cat.categories.astype(str))
    labels = pd.Index([f"{x}|{y}" for x, y in zip(cats_a[uniq // nb], cats_b[uniq % nb])])
    return codes, labels


def as_text(s):
    return s.astype(str).fillna("nan")


def with_extra(main, extra):
    main = pd.Index(main).drop_duplicates()
    return main.append(pd.Index(extra).difference(main))


def codes_of(values, known):
    s = pd.Series(values)
    code = pd.factorize(s.astype(str).where(s.notna()))[0]
    seen = np.zeros(code.max() + 2, dtype=bool)
    seen[code[known & (code >= 0)]] = True
    return np.where((code >= 0) & seen[np.maximum(code, 0)], code, -1)


def standardise(x, known):
    x = np.asarray(x, dtype=float)
    ok = np.isfinite(x)
    ref = x[ok & known]
    mu, sd = (ref.mean(), ref.std() + 1e-6) if len(ref) else (0.0, 1.0)
    return np.where(ok, (x - mu) / sd, 0.0), ok.astype(float)


def nb_nll_t(y, eta, log_r):
    y, eta = y.double(), eta.double().clamp(-ETA_MAX, ETA_MAX)
    lr = log_r.double().clamp(LOG_R_MIN, LOG_R_MAX)
    r = lr.exp()
    lp = (torch.lgamma(y + r) - torch.lgamma(r) - torch.lgamma(y + 1.0) + r * lr + y * eta
          - (y + r) * torch.logaddexp(lr, eta))
    return (-lp).clamp(max=NLL_CAP)


def mlp(n_in, n_out):
    net = nn.Sequential(nn.Linear(n_in, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, HIDDEN), nn.GELU(),
                        nn.Linear(HIDDEN, n_out))
    nn.init.zeros_(net[-1].weight)
    nn.init.zeros_(net[-1].bias)
    return net


def as_t(x, device, dtype=torch.long):
    return torch.as_tensor(np.asarray(x), dtype=dtype, device=device)


def calibrated(P, th):
    eta = P["eta0"] + th[0] * P["inter"] + th[1] + th[2] * P["win"] + th[3] * P["knn"] + th[9] * P["gknn"]
    log_r = th[4] + th[5] * P["log_r"] + th[6] * P["lvg"] + th[7] * P["lnr"] + th[8] * (eta - LEVEL)
    return eta, log_r


def fit_theta(P, y, free, iters=100):
    base = torch.tensor(THETA0, dtype=torch.float64, device=y.device)
    idx = torch.tensor(free, device=y.device)
    p = base[idx].clone().requires_grad_(True)
    opt = torch.optim.LBFGS([p], lr=1.0, max_iter=iters, tolerance_grad=1e-9, tolerance_change=1e-12,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = nb_nll_t(y, *calibrated(P, base.index_put((idx,), p))).mean()
        loss.backward()
        return loss
    opt.step(closure)
    th = base.index_put((idx,), p.detach())
    with torch.no_grad():
        loss = float(nb_nll_t(y, *calibrated(P, th)).mean())
    if not np.isfinite(loss):
        th = base
        loss = float(nb_nll_t(y, *calibrated(P, th)).mean())
    return th, loss


def catboost(classify, X, y, w, seed, **params):
    from catboost import CatBoostClassifier, CatBoostRegressor
    cls = CatBoostClassifier if classify else CatBoostRegressor
    kw = dict(random_seed=seed, verbose=False, allow_writing_files=False, **params)
    try:
        model = cls(task_type="GPU", devices="0", gpu_ram_part=0.3, **kw)
        model.fit(X, y, sample_weight=w)
    except Exception as e:
        log(f"CatBoost on GPU failed ({e!r}); fitting on the CPU")
        model = cls(thread_count=-1, **kw)
        model.fit(X, y, sample_weight=w)
    return model


def boost(X, y, seed, w=None):
    return catboost(False, X, y, w, seed, loss_function="RMSE", depth=6, iterations=STACK_ITERS, learning_rate=0.08)


def domain_weights(D, Dt, seed):
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    a = rng.permutation(len(D))[:SHIFT_SAMPLE]
    b = rng.permutation(len(Dt))[:SHIFT_SAMPLE]
    Z = np.vstack([D[a], Dt[b]])
    lab = np.r_[np.zeros(len(a)), np.ones(len(b))]
    hold = rng.random(len(Z)) < 0.2
    clf = catboost(True, Z[~hold], lab[~hold], None, seed, loss_function="Logloss", depth=6, iterations=SHIFT_ITERS,
                   learning_rate=0.1)
    auc = float("nan")
    if 0 < lab[hold].sum() < hold.sum():
        auc = roc_auc_score(lab[hold], clf.predict_proba(Z[hold])[:, 1])
    p = clf.predict_proba(D)[:, 1].clip(1e-6, 1 - 1e-6)
    prior = (lab[~hold] == 0).sum() / max((lab[~hold] == 1).sum(), 1)
    w = np.clip(p / (1 - p) * prior, *SHIFT_CLIP)
    w = w / w.mean()
    return w, auc, float(w.sum() ** 2 / (w ** 2).sum() / len(w))


def stacked(eta, lr, delta, err, th):
    e = eta + th[0] * delta + th[1]
    return e, th[2] + th[3] * lr + th[4] * err + th[5] * (e - LEVEL)


def fit_stacked(eta, lr, delta, err, y, w):
    t0 = torch.tensor([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=torch.float64, device=y.device)
    p = t0.clone().requires_grad_(True)
    opt = torch.optim.LBFGS([p], lr=1.0, max_iter=100, line_search_fn="strong_wolfe")
    w = w / w.sum()

    def closure():
        opt.zero_grad()
        loss = (w * nb_nll_t(y, *stacked(eta, lr, delta, err, p))).sum()
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        loss = float((w * nb_nll_t(y, *stacked(eta, lr, delta, err, p))).sum())
        base = float((w * nb_nll_t(y, eta, lr)).sum())
    return p.detach(), loss, base


def parse_gbff(path):
    rows, feat, chr_id, chr_name, in_feat = [], None, None, None, False

    def flush():
        if feat is not None and feat["key"] == "CDS":
            loc = feat["loc"]
            nums = [int(x) for x in re.findall(r"\d+", loc)]
            if nums:
                q = feat["q"]
                rows.append((path.stem, chr_id, chr_name, min(nums), "-" if "complement" in loc else "+",
                             q.get("gene", ""), q.get("product", "")))
    with open(path, errors="replace") as fh:
        for line in fh:
            if line.startswith("LOCUS"):
                chr_name = chr_id = line.split()[1]
            elif line.startswith("VERSION"):
                parts = line.split()
                if len(parts) > 1:
                    chr_id = parts[1]
            elif line.startswith("FEATURES"):
                in_feat = True
            elif line.startswith(("ORIGIN", "CONTIG", "//")):
                flush()
                feat, in_feat = None, False
            elif in_feat and len(line) > 21:
                if line[5] != " ":
                    flush()
                    feat = {"key": line[5:21].strip(), "loc": line[21:].strip(), "q": {}, "last": None}
                elif feat is not None:
                    s = line[21:].strip()
                    if s.startswith("/"):
                        k, _, v = s[1:].partition("=")
                        feat["q"][k] = v.strip('"')
                        feat["last"] = k
                    elif feat["last"] is None:
                        feat["loc"] += s
                    else:
                        feat["q"][feat["last"]] = (feat["q"][feat["last"]] + " " + s).strip('"')
    return pd.DataFrame(rows, columns=["genome_id", "chr_id", "chr_name", "start", "strand", "gene", "product"])


def _parse_chunk(paths, conn):
    try:
        conn.send([parse_gbff(p) for p in paths])
    except BaseException as e:
        conn.send(RuntimeError(f"parse_gbff failed: {e!r}"))
    conn.close()


def parse_all(paths):
    n = max(1, min(8, os.cpu_count() or 1, len(paths)))
    try:
        ctx = mp.get_context("fork")
        jobs = []
        for i in range(n):
            recv, send = ctx.Pipe(duplex=False)
            proc = ctx.Process(target=_parse_chunk, args=(paths[i::n], send), daemon=True)
            proc.start()
            send.close()
            jobs.append((proc, recv))
        frames = []
        for proc, recv in jobs:
            got = recv.recv()
            proc.join()
            if isinstance(got, BaseException):
                raise got
            frames += got
    except Exception as e:
        log(f"GenBank: parallel parse failed ({e!r}); parsing serially")
        frames = [parse_gbff(p) for p in paths]
    return pd.concat(frames, ignore_index=True)


def gene_words(f, genomes_dir, known):
    files = {p.stem: p for p in Path(genomes_dir).glob("*.gbff")}
    paths = [files[g] for g in sorted(f["genome_id"].astype(str).unique()) if g in files]
    gb = parse_all(paths)
    gb["start"] = gb["start"].astype(float)
    pos = ["genome_id", "chr", "start", "strand"]
    keys = pd.DataFrame({"genome_id": f["genome_id"].astype(str).to_numpy(),
                         "chr": as_text(f["Chr"]).str.split(";").str[0].to_numpy(),
                         "start": pd.to_numeric(as_text(f["Start"]).str.split(";").str[0], errors="coerce").to_numpy(),
                         "strand": as_text(f["Strand"]).str.split(";").str[0].to_numpy()})
    cols = ["gene", "product"]
    got = keys.merge(gb.rename(columns={"chr_id": "chr"}).drop(columns="chr_name").drop_duplicates(pos),
                     on=pos, how="left")[cols]
    miss = got["product"].isna().to_numpy()
    alt = keys[miss].merge(gb.rename(columns={"chr_name": "chr"}).drop(columns="chr_id").drop_duplicates(pos),
                           on=pos, how="left")[cols]
    got.loc[miss, cols] = alt.to_numpy()
    got = got.fillna("")
    sym = got["gene"].str.lower()
    toks = [re.findall(r"[a-z0-9]{2,}", p.lower()) + ([f"sym_{s}", f"pre_{s[:3]}"] if s else [])
            for p, s in zip(got["product"], sym)]
    count = {}
    for t, k in zip(toks, known):
        if k:
            for w in set(t):
                count[w] = count.get(w, 0) + 1
    vocab = {w: i for i, w in enumerate(sorted(w for w, c in count.items() if c >= MIN_WORD_GENES))}
    idx = [[vocab[w] for w in t if w in vocab] for t in toks]
    matched = ~miss
    matched[miss] = alt["product"].notna().to_numpy()
    log(f"GenBank: {len(paths)} files, {len(gb):,} CDS; genes matched {matched.mean():.4f}; vocabulary {len(vocab):,}; genes with a word {np.mean([len(i) > 0 for i in idx]):.3f}")
    return idx, len(vocab)


def shipped_units(path, index):
    keys = pd.read_parquet(Path(path) / "gene_model_keys.parquet")
    pos = pd.Index(as_text(keys["genome_id"]) + "|" + as_text(keys["feature_id"])).get_indexer(index)
    unit = np.where(pos >= 0, keys["unit"].to_numpy(dtype=np.int64)[np.maximum(pos, 0)], -1)
    own = unit < 0
    unit[own] = unit.max() + 1 + np.arange(own.sum())
    return pd.factorize(unit)[0]


def shipped(path, keys_name, values_name, index, key_cols, known, flag=None):
    keys = pd.read_parquet(Path(path) / keys_name)
    emb = np.load(Path(path) / values_name).astype(np.float32)
    lab = keys[key_cols[0]].astype(str) + "|" + keys[key_cols[1]].astype(str)
    pos = pd.Index(lab).get_indexer(index)
    ok = pos >= 0
    if flag is not None:
        ok[ok] = keys[flag].to_numpy(dtype=bool)[pos[ok]]
    out = np.zeros((len(index), emb.shape[1]), dtype=np.float32)
    out[ok] = emb[pos[ok]]
    ref = out[ok & known]
    out[ok] = (out[ok] - ref.mean(0)) / (ref.std(0) + 1e-6)
    log(f"{values_name}: {emb.shape[1]} dims, rows present {ok.mean():.4f}")
    return np.column_stack([out, ok.astype(np.float32)])


class Net(nn.Module):

    def __init__(self, gs, rs, base, a0, device):
        super().__init__()
        D = RANK + 2
        self.gfree = nn.Parameter(torch.zeros(gs["n"] + 1, D))
        self.rfree = nn.Parameter(torch.zeros(rs["n"] + 1, D))
        with torch.no_grad():
            self.gfree[:, 1:RANK + 1].normal_(0.0, 0.01)
            self.rfree[:, 1:RANK + 1].normal_(0.0, 0.01)
            self.gfree[:-1, 0] = torch.as_tensor(a0, dtype=torch.float32)
        self.g_genome = nn.Embedding(gs["n_genome"] + 1, EMB_GENOME)
        self.r_genome = nn.Embedding(rs["n_genome"] + 1, EMB_GENOME)
        self.r_cats = nn.ModuleList(nn.Embedding(n + 1, EMB_CAT) for n in rs["cat_sizes"])
        g_in = 3 * (D + 1) + EMB_GENOME + gs["num"].shape[1]
        self.words = nn.EmbeddingBag(gs["n_words"] + 1, EMB_WORD, mode="mean")
        nn.init.normal_(self.words.weight, 0.0, 0.1)
        g_in += EMB_WORD + gs["esm"].shape[1] + gs["gm"].shape[1]
        r_in = (len(RUN_GROUPS) * (D + 1) + EMB_GENOME + EMB_CAT * len(rs["cat_sizes"]) + rs["num"].shape[1]
                + rs["text"].shape[1])
        self.g_tower = mlp(g_in, D)
        self.r_tower = mlp(r_in, D)
        self.base = nn.Parameter(torch.tensor(float(base)))
        self.kind_e = nn.Parameter(torch.zeros(4))
        self.kind_f = nn.Parameter(torch.zeros(4))
        self.to(device)

    @staticmethod
    def loo(F, keep, grp, n_groups):
        w = keep * (grp < n_groups).float()
        S = torch.zeros(n_groups + 1, F.shape[1], device=F.device).index_add_(0, grp, F * w[:, None])
        n = torch.zeros(n_groups + 1, device=F.device).index_add_(0, grp, w)
        m = n[grp] - w
        has = (m > 0).float()
        mean = (S[grp] - F * w[:, None]) / m.clamp(min=1.0)[:, None] * has[:, None]
        return torch.cat([mean, has[:, None]], 1)

    def tower(self, net, parts, device):
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            return net(torch.cat(parts, 1)).float()

    def gene_tower(self, gs, keep):
        F = (self.gfree * keep[:, None]).detach()
        cnt = keep[gs["prv"]] + keep[gs["nxt"]]
        nb = (F[gs["prv"]] + F[gs["nxt"]]) / cnt.clamp(min=1.0)[:, None]
        parts = [self.loo(F, keep, gs["hom"], gs["n_hom"]), self.loo(F, keep, gs["op"], gs["n_op"]),
                 nb, (cnt > 0).float()[:, None], self.g_genome(gs["genome"]), gs["num"],
                 self.words(gs["w_idx"], gs["w_off"]), gs["esm"], gs["gm"]]
        return self.tower(self.g_tower, parts, F.device)

    def run_tower(self, rs, keep):
        F = (self.rfree * keep[:, None]).detach()
        parts = [self.loo(F, keep, grp, n) for grp, n in zip(rs["groups"], rs["n_groups"])]
        parts += [self.r_genome(rs["genome"])] + [emb(c) for emb, c in zip(self.r_cats, rs["cats"])]
        parts += [rs["num"], rs["text"]]
        return self.tower(self.r_tower, parts, F.device)

    def tables(self, gs, rs, keep_g, keep_r):
        return (self.gene_tower(gs, keep_g) + self.gfree * keep_g[:, None],
                self.run_tower(rs, keep_r) + self.rfree * keep_r[:, None])

    def cells(self, Gt, Rt, g, r, off, kind):
        a, u, c = Gt[g, 0], Gt[g, 1:RANK + 1], Gt[g, RANK + 1]
        b, v, d = Rt[r, 0], Rt[r, 1:RANK + 1], Rt[r, RANK + 1]
        eta0 = off + self.base + a + b
        inter = (u * v).sum(1)
        log_r = c + d + self.kind_e[kind] + self.kind_f[kind] * (eta0 + inter - LEVEL)
        return eta0, inter, log_r


class Member:

    def __init__(self, net):
        self.net, self.theta = net, {}


class Model:

    def __init__(self, feat, sub_dir):
        self.feat, self.sub_dir = feat, Path(sub_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.members, self.tau = [], {k: 0.0 for k in KINDS}
        self.stack, self.stack_data = {}, {}
        log(f"arm {ARM}: pseudo-A with operons {PSEUDO_A_OPERON}, operon neighbours {USE_OPERON}, cross-genome "
            f"neighbours {USE_XKNN}, domain weights {USE_SHIFT}, gene model {GENE_MODEL}, seeds {SEEDS}")


    def gene_static(self, known, genome_code, genomes_dir):
        G, dev = len(self.genes), self.device
        f = self.feat.copy()
        f.index = f["genome_id"].astype(str) + "|" + f["feature_id"].astype(str)
        f = f[~f.index.duplicated()].reindex(self.genes)
        f["genome_id"] = self.genes.str.split("|").str[0]
        words = gene_words(f, genomes_dir, known)

        def piece(col, i):
            return as_text(f[col]).str.split(";").str[i]
        start = pd.to_numeric(piece("Start", 0), errors="coerce").to_numpy(dtype=float)
        end = pd.to_numeric(piece("End", -1), errors="coerce").to_numpy(dtype=float)
        strand = piece("Strand", 0).to_numpy(dtype=str)
        chrom = pd.factorize(pd.Series(genome_code, index=f.index).astype(str) + "|" + piece("Chr", 0))[0]
        ok = np.flatnonzero(np.isfinite(start) & f["Chr"].notna().to_numpy())
        o = ok[np.lexsort((start[ok], chrom[ok]))]
        same = chrom[o[1:]] == chrom[o[:-1]]
        prv, nxt = np.full(G, G), np.full(G, G)
        nxt[o[:-1][same]] = o[1:][same]
        prv[o[1:][same]] = o[:-1][same]
        gap = start[o[1:]] - end[o[:-1]]
        same_strand = same & (strand[o[1:]] == strand[o[:-1]])

        def operons(link):
            starts = np.r_[True, ~link]
            lab = np.arange(G) + G
            lab[o] = np.cumsum(starts) - 1
            return pd.factorize(lab)[0], starts
        op, starts = operons(same_strand & (gap < OPERON_GAP))
        op30, _ = operons(same_strand & (gap < PSEUDO_A_OPERON_GAP))
        op_size = np.bincount(op)[op]
        first = np.zeros(G)
        first[o[starts]] = 1.0
        side = {k: np.zeros(G) for k in ("same_prv", "same_nxt")}
        side.update(gap_prv=np.full(G, 3.0), gap_nxt=np.full(G, 3.0))
        side["same_prv"][o[1:][same]] = same_strand[same]
        side["same_nxt"][o[:-1][same]] = same_strand[same]
        side["gap_prv"][o[1:][same]] = np.clip(gap[same], -300.0, 3000.0) / 1000.0
        side["gap_nxt"][o[:-1][same]] = np.clip(gap[same], -300.0, 3000.0) / 1000.0
        hom = pd.factorize((as_text(f["ref_genome_id"]) + "|" + as_text(f["ref_feature_id"])).where(
            f["ref_feature_id"].notna()))[0]
        n_hom, n_op = int(hom.max()) + 1, int(op.max()) + 1
        unit_c = shipped_units(self.sub_dir, self.genes)
        unit_a = pd.factorize(np.where(hom >= 0, hom, n_hom + op))[0]
        length = pd.to_numeric(f["Length"], errors="coerce").to_numpy(dtype=float)
        loglen, has_len = standardise(np.log(np.clip(length, 1.0, None)), known)
        lop, _ = standardise(np.log(op_size), known)
        num = np.column_stack([loglen, has_len, lop, first])
        log(f"genes {G:,} (fit-known {known.sum():,}): position {len(ok) / G:.3f}, operons {n_op:,} (genes in a "
            f"multi-gene operon {np.mean(op_size > 1):.3f}), genes with a homolog key {np.mean(hom >= 0):.3f}, "
            f"C units {unit_c.max() + 1:,}")
        gs = dict(
            n=G, n_genome=int(genome_code.max()) + 1, n_hom=n_hom, n_op=n_op, n_unit_c=int(unit_c.max()) + 2,
            unit_c_np=unit_c, unit_a_np=unit_a, has_hom_np=(hom >= 0).astype(float), hom_np=hom, op30_np=op30,
            nxt_np=nxt, genome_np=genome_code,
            **{k: as_t(np.append(v, 3.0 if k.startswith("gap") else 0.0), dev, torch.float32) for k, v in side.items()},
            known=as_t(np.append(known, False), dev, torch.float32),
            hom=as_t(np.append(np.where(hom >= 0, hom, n_hom), n_hom), dev),
            op=as_t(np.append(op, n_op), dev),
            unit_c=as_t(np.append(unit_c, unit_c.max() + 1), dev),
            prv=as_t(np.append(prv, G), dev), nxt=as_t(np.append(nxt, G), dev),
            genome=as_t(np.append(genome_code, genome_code.max() + 1), dev),
            num=as_t(np.vstack([num, np.zeros((1, num.shape[1]))]), dev, torch.float32))
        idx, n_words = words
        flat = np.fromiter((i for lst in idx for i in lst), dtype=np.int64)
        offsets = np.r_[0, np.cumsum([len(lst) for lst in idx])]
        gs.update(n_words=n_words, w_idx=as_t(flat, dev), w_off=as_t(offsets, dev))
        esm = shipped(self.sub_dir, "gene_keys.parquet", "gene_esm.npy", self.genes, ["genome_id", "feature_id"],
                      known, flag="has_esm")
        gs["esm"] = as_t(np.vstack([esm, np.zeros((1, esm.shape[1]))]), dev, torch.float32)
        gm = shipped(self.sub_dir, "gene_model_keys.parquet", f"gene_model_{GENE_MODEL}.npy", self.genes,
                     ["genome_id", "feature_id"], known)
        gs["gm"] = as_t(np.vstack([gm, np.zeros((1, gm.shape[1]))]), dev, torch.float32)
        gs["gm_level"] = gm[:, [0, 1, -1]]
        return gs

    def run_static(self, samples_df, known, genome_code):
        R, dev = len(self.runs), self.device
        s = samples_df.copy()
        s.index = s["genome_id"].astype(str) + "|" + s["run_accession"].astype(str)
        s = s[~s.index.duplicated()].reindex(self.runs)
        sra = pd.read_parquet(self.sub_dir / "sra_runs.parquet")
        sra.index = as_text(sra["genome_id"]) + "|" + as_text(sra["run_accession"])
        sra = sra[~sra.index.duplicated()].reindex(self.runs)
        sra["log_read_length"] = np.log(pd.to_numeric(sra["read_length"], errors="coerce").clip(lower=1.0))
        for k in SRA_NUMS[1:]:
            sra[k] = pd.to_numeric(sra[k].astype(float) if sra[k].dtype == bool else sra[k], errors="coerce")
        s = s.join(sra[list(SRA_CATS + SRA_NUMS)])
        log(f"SRA metadata: runs found {sra['instrument_platform'].notna().mean():.3f}")
        run_cats, run_nums = RUN_CATS + SRA_CATS, RUN_NUMS + SRA_NUMS
        groups, n_groups = [], []
        for k in RUN_GROUPS:
            c = codes_of(s[k].to_numpy(), known) if k in s else np.full(R, -1)
            n = int(c.max()) + 1
            groups.append(as_t(np.append(np.where(c >= 0, c, n), n), dev))
            n_groups.append(n)
        cats, cat_sizes = [], []
        for k in run_cats:
            c = codes_of(s[k].to_numpy(), known) + 1 if k in s else np.zeros(R, dtype=np.int64)
            cats.append(as_t(np.append(c, 0), dev))
            cat_sizes.append(int(c.max()) + 1)
        nums = [standardise(pd.to_numeric(s[k], errors="coerce"), known) if k in s else (np.zeros(R), np.zeros(R))
                for k in run_nums]
        num = np.column_stack([x for pair in nums for x in pair])
        genome = pd.Series(self.runs.str.split("|").str[0], index=s.index)
        prj_label = s["prj_accession"].astype(str).where(s["prj_accession"].notna(), s.index.to_series()) \
            if "prj_accession" in s else s.index.to_series()
        prj = pd.factorize(genome + "|" + prj_label)[0]
        cond = genome + "|" + prj_label
        for k in CONDITION:
            if k in s:
                cond = cond + "|" + as_text(s[k])
        rep = pd.factorize(cond)[0]
        log(f"runs {R:,} (fit-known {known.sum():,}): projects {prj.max() + 1:,}, replicate groups {rep.max() + 1:,}; "
            f"groups with a known run " + ", ".join(f"{k} {n:,}" for k, n in zip(RUN_GROUPS, n_groups)))
        rs = dict(
            n=R, n_genome=int(genome_code.max()) + 1, n_groups=n_groups, cat_sizes=cat_sizes, groups=groups, cats=cats,
            prj_np=prj, rep_np=rep, genome_np=genome_code, n_prj=int(prj.max()) + 2, n_rep=int(rep.max()) + 2,
            known=as_t(np.append(known, False), dev, torch.float32),
            prj=as_t(np.append(prj, prj.max() + 1), dev), rep=as_t(np.append(rep, rep.max() + 1), dev),
            genome=as_t(np.append(genome_code, genome_code.max() + 1), dev),
            num=as_t(np.vstack([num, np.zeros((1, num.shape[1]))]), dev, torch.float32),
            lsf=pd.to_numeric(s["log_size_factor"], errors="coerce").fillna(0.0).to_numpy(),
            sra=s[["log_read_length", "year"]].astype(float).fillna(0.0).to_numpy())
        text = shipped(self.sub_dir, "run_keys.parquet", "run_text.npy", self.runs, ["genome_id", "run_accession"],
                       known)
        rs["text"] = as_t(np.vstack([text, np.zeros((1, text.shape[1]))]), dev, torch.float32)
        return rs


    def pseudo_split(self, gi, ri, gs, rs):
        rng = np.random.default_rng(SPLIT_SEED)
        G, R = len(self.genes), len(self.runs)
        g_seen = np.bincount(gi, minlength=G) > 0
        r_seen = np.bincount(ri, minlength=R) > 0
        unit_c, unit_a = gs["unit_c_np"], gs["unit_a_np"]
        pc = (rng.random(unit_c.max() + 1) < PSEUDO_C_UNITS)[unit_c] & g_seen
        if not pc.any():
            pc = (unit_c == unit_c[rng.choice(np.flatnonzero(g_seen))]) & g_seen
        prj, rep = rs["prj_np"], rs["rep_np"]
        prj_size = np.bincount(prj[r_seen], minlength=prj.max() + 1)
        rep_size = np.bincount(rep[r_seen], minlength=rep.max() + 1)
        pick_p = (rng.random(prj.max() + 1) < PSEUDO_B_PROJECTS) & (prj_size <= PSEUDO_B_MAX_RUNS)
        pick_r = (rng.random(rep.max() + 1) < PSEUDO_B_GROUPS) & (rep_size <= PSEUDO_B_MAX_RUNS)
        pb = (pick_p[prj] | pick_r[rep]) & r_seen
        if not pb.any():
            pb = (rep == rep[rng.choice(np.flatnonzero(r_seen))]) & r_seen
        pa_run = np.zeros(R, dtype=bool)
        cand = r_seen & ~pb
        for k in np.unique(rs["genome_np"][cand]):
            runs = np.flatnonzero(cand & (rs["genome_np"] == k))
            if len(runs) < PSEUDO_A_MIN_RUNS:
                continue
            taken = 0
            for p in rng.permutation(np.unique(prj[runs])):
                members = runs[prj[runs] == p]
                if len(members) > 0.5 * len(runs):
                    continue
                pa_run[members] = True
                taken += len(members)
                if taken >= PSEUDO_A_RUNS * len(runs):
                    break
        if PSEUDO_A_OPERON:
            pa_gene = self.pseudo_a_genes(rng, gs, pa_run, rs["genome_np"], g_seen & ~pc)
        else:
            pa_gene = (rng.random(unit_a.max() + 1) < PSEUDO_A_UNITS)[unit_a] & g_seen & ~pc
        held = pc[gi] | pb[ri] | (pa_gene[gi] & pa_run[ri])
        g_known = np.bincount(gi[~held], minlength=G) > 0
        r_known = np.bincount(ri[~held], minlength=R) > 0
        kind = np.where(held, (~r_known[ri]).astype(np.int8) + 2 * (~g_known[gi]).astype(np.int8), -1).astype(np.int8)
        log(f"pseudo-regions: C genes {pc.sum():,}, B runs {pb.sum():,}, A runs {pa_run.sum():,} x A genes "
            f"{pa_gene.sum():,}; cells " + ", ".join(f"{n} {np.sum(kind == k):,}" for k, n in enumerate(KINDS))
            + f", fit {np.sum(kind < 0):,}")
        return kind, g_known, r_known

    @staticmethod
    def pseudo_a_genes(rng, gs, pa_run, run_genome, eligible):
        hom, op30, genome = gs["hom_np"], gs["op30_np"], gs["genome_np"]
        seed_unit = pd.factorize(np.where(hom >= 0, hom, hom.max() + 1 + op30))[0]
        u = rng.random(seed_unit.max() + 1)[seed_unit]
        comp = np.ones(op30.max() + 1)
        np.minimum.at(comp, op30, u)
        draw = comp[op30]
        in_a = np.isin(genome, np.unique(run_genome[pa_run]))
        ref = draw[eligible & in_a]
        p = float(np.quantile(ref, PSEUDO_A_GENE_SHARE)) if len(ref) else 0.0
        pa = (draw <= p) & eligible
        nxt = gs["nxt_np"]
        ok = (nxt < len(nxt)) & eligible & in_a
        co = pa[nxt[ok & pa]].mean() if (ok & pa).any() else float("nan")
        bg = pa[nxt[ok & ~pa]].mean() if (ok & ~pa).any() else float("nan")
        log(f"pseudo-A genes: threshold {p:.3f}, share of eligible genes in pseudo-A genomes "
            f"{pa[eligible & in_a].mean() if (eligible & in_a).any() else 0.0:.3f}; next gene also pseudo-A {co:.3f} "
            f"(when the gene is not: {bg:.3f})")
        return pa


    def fit(self, train_df, samples_df, genomes_dir):
        t_fit = time.time()
        y = train_df["readcount"].to_numpy(dtype=np.float64)
        off = (train_df["log_length_ratio"].to_numpy(dtype=np.float64)
               + train_df["log_size_factor"].to_numpy(dtype=np.float64))
        g_codes, g_train = pair_index(train_df, "genome_id", "feature_id")
        r_codes, r_train = pair_index(train_df, "genome_id", "run_accession")
        self.genes = with_extra(self.feat["genome_id"].astype(str) + "|" + self.feat["feature_id"].astype(str), g_train)
        self.runs = with_extra(samples_df["genome_id"].astype(str) + "|" + samples_df["run_accession"].astype(str),
                               r_train)
        gi = self.genes.get_indexer(g_train)[g_codes].astype(np.int64)
        ri = self.runs.get_indexer(r_train)[r_codes].astype(np.int64)
        del g_codes, r_codes
        G, R = len(self.genes), len(self.runs)
        genomes = pd.Index(sorted(set(self.genes.str.split("|").str[0]) | set(self.runs.str.split("|").str[0])))
        g_code = genomes.get_indexer(self.genes.str.split("|").str[0])
        r_code = genomes.get_indexer(self.runs.str.split("|").str[0])
        log(f"train rows {len(y):,}, genes {G:,}, runs {R:,}, genomes {len(genomes)}, device {self.device}")

        all_g = np.bincount(gi, minlength=G) > 0
        all_r = np.bincount(ri, minlength=R) > 0
        gs0 = self.gene_static(all_g, g_code, genomes_dir)
        rs0 = self.run_static(samples_df, all_r, r_code)
        kind, g_known, r_known = self.pseudo_split(gi, ri, gs0, rs0)
        self.gs, self.rs = gs0, rs0
        self.gs["known"] = as_t(np.append(g_known, False), self.device, torch.float32)
        self.rs["known"] = as_t(np.append(r_known, False), self.device, torch.float32)
        self.absorb_g = as_t(np.append(all_g & ~g_known, False), self.device, torch.float32)
        self.absorb_r = as_t(np.append(all_r & ~r_known, False), self.device, torch.float32)

        in_fit = kind < 0
        e = np.exp(off[in_fit])
        base = float(np.log(y[in_fit].sum() / e.sum()))
        sg = np.bincount(gi[in_fit], weights=y[in_fit], minlength=G)
        eg = np.bincount(gi[in_fit], weights=e, minlength=G)
        a0 = np.where(g_known, np.log((sg + np.exp(base)) / np.where(g_known, eg + 1.0, 1.0)) - base, 0.0)
        del e, sg, eg
        dev = self.device
        rng = np.random.default_rng(SPLIT_SEED + 1)
        self.pseudo = {}
        for k, name in enumerate(KINDS):
            rows = np.flatnonzero(kind == k)
            sub = rng.permutation(rows)[:CALIB_CELLS]
            self.pseudo[name] = dict(g=as_t(gi[sub], dev), r=as_t(ri[sub], dev), off=as_t(off[sub], dev, torch.float32),
                                     y=as_t(y[sub], dev, torch.float64))
        bcd = np.flatnonzero(kind >= 1)
        self.absorb_cells = dict(g=as_t(gi[bcd], dev), r=as_t(ri[bcd], dev), off=as_t(off[bcd], dev, torch.float32),
                                 y=as_t(y[bcd], dev, torch.float32))
        self.meta_tables(gi, ri, g_code, r_code)
        meta_final = (self.g_meta, self.r_meta)
        self.meta_tables(gi[in_fit], ri[in_fit], g_code, r_code)
        self.Y, self.OFF = as_t(y[in_fit], dev, torch.float32), as_t(off[in_fit], dev, torch.float32)
        self.GI, self.RI = as_t(gi[in_fit], dev, torch.int32), as_t(ri[in_fit], dev, torch.int32)
        del y, off, gi, ri, kind, in_fit
        if USE_XKNN:
            self.xknn_tables()
        self.neighbour_index()

        for i, seed in enumerate(SEEDS):
            budget = (FIT_BUDGET - STACK_RESERVE - (time.time() - t_fit)) / (len(SEEDS) - i) - 60.0
            if self.members and budget < MIN_SEED_SECONDS:
                log(f"seed {seed}: {budget:.0f}s left per seed, ensemble stops at {len(self.members)} members")
                break
            member = self.train_member(seed, base, a0, budget)
            self.calibrate(member)
            self.members.append(member)
        if len(self.members) > 1:
            self.fit_tau()
        self.stack = {}
        self.stack_inputs()
        self.g_meta, self.r_meta = meta_final
        self.absorb()
        d = self.absorb_cells
        self.Y, self.OFF = torch.cat([self.Y, d["y"]]), torch.cat([self.OFF, d["off"]])
        self.GI, self.RI = torch.cat([self.GI, d["g"].int()]), torch.cat([self.RI, d["r"].int()])
        del self.absorb_cells, self.pseudo, self.key_sorted, self.key_order, self.by_genome
        torch.cuda.empty_cache()
        self.neighbour_index()
        for member in self.members:
            self.residual_features(member)
        del self.Y, self.OFF, self.GI, self.RI
        del self.by_genome
        log(f"fit done in {time.time() - t_fit:.0f}s")
        return self


    def train_member(self, seed, base, a0, budget):
        gs, rs, dev = self.gs, self.rs, self.device
        torch.manual_seed(seed)
        net = Net(gs, rs, base, a0, dev)
        towers = [p for name, p in net.named_parameters() if name not in ("gfree", "rfree")]
        opt = torch.optim.Adam([{"params": [net.gfree, net.rfree], "lr": LR_FREE, "weight_decay": 0.0},
                                {"params": towers, "lr": LR_NET, "weight_decay": WD_NET}])
        base_lr = [LR_FREE, LR_NET]
        scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
        gen = torch.Generator(device=dev).manual_seed(seed)
        n = len(self.Y)
        batch = min(BATCH, n)
        t, avg, best = time.time(), None, (np.inf, None, 0)
        log(f"seed {seed}: training, budget {budget:.0f}s")
        for step in range(1, STEPS + 1):
            scale = step / WARMUP if step <= WARMUP else \
                0.1 + 0.45 * (1.0 + np.cos(np.pi * (step - WARMUP) / max(STEPS - WARMUP, 1)))
            for grp, lr in zip(opt.param_groups, base_lr):
                grp["lr"] = lr * scale
            drop_g = (torch.rand(gs["n_unit_c"], device=dev, generator=gen) < P_DROP)[gs["unit_c"]]
            drop_r = ((torch.rand(rs["n_prj"], device=dev, generator=gen) < P_DROP / 2)[rs["prj"]]
                      | (torch.rand(rs["n_rep"], device=dev, generator=gen) < P_DROP / 2)[rs["rep"]])
            keep_g = gs["known"] * (~drop_g).float()
            keep_r = rs["known"] * (~drop_r).float()
            Gt, Rt = net.tables(gs, rs, keep_g, keep_r)
            idx = torch.randint(n, (batch,), device=dev, generator=gen)
            g, r = self.GI[idx].long(), self.RI[idx].long()
            kind = drop_r[r].long() + 2 * drop_g[g].long()
            eta0, inter, log_r = net.cells(Gt, Rt, g, r, self.OFF[idx], kind)
            nll = nb_nll_t(self.Y[idx], eta0 + inter, log_r).mean()
            reg = L2_FREE * ((net.gfree[g, 1:RANK + 1] ** 2).sum(1).mean()
                             + (net.rfree[r, 1:RANK + 1] ** 2).sum(1).mean())
            opt.zero_grad(set_to_none=True)
            scaler.scale(nll + reg).backward()
            scaler.step(opt)
            scaler.update()
            v = nll.item()
            avg = v if avg is None else 0.98 * avg + 0.02 * v
            late = time.time() - t > budget
            if step % 100 == 0:
                log(f"seed {seed} step {step}/{STEPS} train NLL (ema) {avg:.4f} lr x{scale:.2f} | "
                    f"{(time.time() - t) / step * 1000:.0f} ms/step")
            if step % EVAL_EVERY == 0 or step == STEPS or late:
                score, per = self.check(net)
                mark = ""
                if score < best[0]:
                    best = (score, {k: x.detach().clone() for k, x in net.state_dict().items()}, step)
                    mark = " *"
                log(f"seed {seed} step {step}: pseudo-region NLL (calibrated) " +
                    " ".join(f"{k} {x:.4f}" for k, x in per.items()) + f" | mean A,B,C {score:.4f}{mark}")
            if late:
                log(f"seed {seed}: time budget reached at step {step}")
                break
        net.load_state_dict(best[1])
        log(f"seed {seed}: kept step {best[2]} (pseudo NLL {best[0]:.4f}); kind e "
            f"{np.round(net.kind_e.detach().cpu().numpy(), 3).tolist()} f {np.round(net.kind_f.detach().cpu().numpy(), 3).tolist()}")
        member = Member(net)
        self.finalise(member)
        return member

    @torch.no_grad()
    def finalise(self, member):
        member.Gt, member.Rt = member.net.tables(self.gs, self.rs, self.gs["known"], self.rs["known"])

    def check(self, net):
        member = Member(net)
        self.finalise(member)
        per = {}
        for name in ("A", "B", "C"):
            d = self.pseudo[name]
            if len(d["y"]) < MIN_CALIB:
                continue
            m = min(CHECK_CELLS, len(d["y"]))
            P = self.parts(member, d["g"][:m], d["r"][:m], d["off"][:m], neigh=False)
            per[name] = fit_theta(P, d["y"][:m], BASIC, iters=50)[1]
        return (float(np.mean(list(per.values()))) if per else 0.0), per


    def neighbour_index(self):
        dev, R1 = self.device, len(self.runs) + 1
        key = self.GI.long() * R1 + self.RI.long()
        self.key_sorted, self.key_order = torch.sort(key)
        del key
        gen_of = self.gs["genome"][self.GI.long()]
        self.by_genome = torch.argsort(gen_of)
        counts = torch.bincount(gen_of, minlength=self.gs["n_genome"] + 1)
        self.genome_bounds = torch.cat([torch.zeros(1, dtype=torch.long, device=dev), counts.cumsum(0)]).tolist()
        del gen_of
        if USE_OPERON or USE_XKNN:
            n = self.gs["n"] + 1
            gi = self.GI.long()
            S = torch.zeros(n, device=dev).index_add_(0, gi, torch.log(self.Y + 0.5) - self.OFF)
            self.gcnt = torch.zeros(n, device=dev).index_add_(0, gi, torch.ones_like(self.Y))
            self.glvl = S / self.gcnt.clamp(min=1.0)
        if USE_XKNN:
            has = self.gcnt > 0
            gen = self.gs["genome"]
            n_gen = self.gs["n_genome"] + 1
            gsum = torch.zeros(n_gen, device=dev).index_add_(0, gen, self.glvl * has)
            gn = torch.zeros(n_gen, device=dev).index_add_(0, gen, has.float())
            gmean = gsum / gn.clamp(min=1.0)
            rel = (self.glvl - gmean[gen]) * has
            w = self.xwts * has.float()[self.xnbr]
            self.xw, top = w.sum(1), w.argmax(1)
            self.xtop = w.gather(1, top[:, None]).squeeze(1)
            self.xlvl = gmean[gen] + (w * rel[self.xnbr]).sum(1) / self.xw.clamp(min=1e-6)
            self.xlvl1 = gmean[gen] + rel[self.xnbr.gather(1, top[:, None]).squeeze(1)]
            vis = self.xw[:-1] > 0
            log(f"cross-genome neighbours: genes with a visible one {float(vis.float().mean()):.3f}, mean top "
                f"similarity {float(self.xtop[:-1][vis].mean()) if vis.any() else 0.0:.3f}")

    def xknn_tables(self):
        keys = pd.read_parquet(self.sub_dir / "gene_model_keys.parquet")
        nbr, sim = np.load(self.sub_dir / "xknn_nbr.npy"), np.load(self.sub_dir / "xknn_sim.npy").astype(np.float32)
        assert len(nbr) == len(keys), "xknn_nbr.npy does not follow gene_model_keys.parquet"
        G = self.gs["n"]
        pos = pd.Index(as_text(keys["genome_id"]) + "|" + as_text(keys["feature_id"])).get_indexer(self.genes)
        to_self = np.full(len(keys) + 1, G, dtype=np.int64)
        to_self[pos[pos >= 0]] = np.flatnonzero(pos >= 0)
        xnbr = np.full((G + 1, nbr.shape[1]), G, dtype=np.int64)
        xwts = np.zeros((G + 1, nbr.shape[1]), dtype=np.float32)
        ok = pos >= 0
        xnbr[:G][ok] = to_self[nbr[pos[ok]]]
        xwts[:G][ok] = np.clip(sim[pos[ok]], 0.0, None) * (xnbr[:G][ok] < G)
        self.xnbr, self.xwts = as_t(xnbr, self.device), as_t(xwts, self.device, torch.float32)
        log(f"cross-genome neighbours: genes with one {np.mean(xwts[:G, 0] > 0):.3f}, mean top-1 similarity "
            f"{xwts[:G, 0][xwts[:G, 0] > 0].mean() if (xwts[:G, 0] > 0).any() else 0.0:.3f}")

    @torch.no_grad()
    def residual_features(self, member):
        gs, rs, dev = self.gs, self.rs, self.device
        n = len(self.Y)
        res = torch.empty(n, device=dev)
        for i in range(0, n, CHUNK):
            g, r = self.GI[i:i + CHUNK].long(), self.RI[i:i + CHUNK].long()
            eta0, inter, _ = member.net.cells(member.Gt, member.Rt, g, r, self.OFF[i:i + CHUNK],
                                              torch.zeros_like(g))
            res[i:i + CHUNK] = torch.log((self.Y[i:i + CHUNK] + 0.5) / (torch.exp(eta0 + inter) + 0.5))
        r2 = res.double() ** 2
        v0 = float(r2.mean())

        def log_var(idx, size):
            s = torch.zeros(size, dtype=torch.float64, device=dev).index_add_(0, idx, r2)
            c = torch.zeros(size, dtype=torch.float64, device=dev).index_add_(0, idx, torch.ones_like(r2))
            return torch.where(c > 0, torch.log((s + 5.0 * v0) / (c + 5.0) / v0), torch.zeros_like(s)).float(), c > 0
        member.lvg, _ = log_var(self.GI.long(), gs["n"] + 1)
        lnr, has = log_var(self.RI.long(), rs["n"] + 1)
        prj = rs["prj"]
        ps = torch.zeros(rs["n_prj"], device=dev).index_add_(0, prj, lnr * has)
        pn = torch.zeros(rs["n_prj"], device=dev).index_add_(0, prj, has.float())
        member.lnr = torch.where(has, lnr, torch.where(pn[prj] > 0, ps[prj] / pn[prj].clamp(min=1), 0.0))
        log(f"residuals: v0 {v0:.3f}, gene log-variability sd {float(member.lvg.std()):.3f}, run log-noise sd "
            f"{float(member.lnr.std()):.3f}")
        member.res_sorted = res[self.key_order]
        nbr = torch.full((rs["n"] + 1, KNN_K), rs["n"], dtype=torch.long, device=dev)
        wts = torch.zeros((rs["n"] + 1, KNN_K), device=dev)
        gnbr = torch.full((gs["n"] + 1, KNN_G), gs["n"], dtype=torch.long, device=dev)
        gwts = torch.zeros((gs["n"] + 1, KNN_G), device=dev)
        b = self.genome_bounds
        for k in range(len(b) - 1):
            rows = self.by_genome[b[k]:b[k + 1]]
            if len(rows) == 0:
                continue
            g_loc, ig = torch.unique(self.GI[rows].long(), return_inverse=True)
            r_loc, ir = torch.unique(self.RI[rows].long(), return_inverse=True)
            kk = min(KNN_K, len(r_loc) - 1)
            if kk < 1:
                continue
            X = torch.zeros(len(g_loc), len(r_loc), device=dev)
            X[ig, ir] = res[rows]
            nrm = X.norm(dim=0) + 1e-6
            S = (X.T @ X) / (nrm[:, None] * nrm[None, :])
            S.fill_diagonal_(-float("inf"))
            val, pos = S.topk(kk, dim=1)
            nbr[r_loc, :kk] = r_loc[pos]
            wts[r_loc, :kk] = val.clamp(min=0.0)
            kg = min(KNN_G, len(g_loc) - 1)
            if kg >= 1:
                gn = X.norm(dim=1) + 1e-6
                Sg = (X @ X.T) / (gn[:, None] * gn[None, :])
                Sg.fill_diagonal_(-float("inf"))
                val, pos = Sg.topk(kg, dim=1)
                gnbr[g_loc, :kg] = g_loc[pos]
                gwts[g_loc, :kg] = val.clamp(min=0.0)
                del Sg
        member.nbr, member.wts, member.gnbr, member.gwts = nbr, wts, gnbr, gwts
        log(f"neighbour runs: mean top-{KNN_K} similarity {float(wts[wts > 0].mean()):.3f}"
            + f"; neighbour genes: mean top-{KNN_G} similarity {float(gwts[gwts > 0].mean()):.3f}")

    def lookup(self, member, g, r):
        key = g * (len(self.runs) + 1) + r
        pos = torch.searchsorted(self.key_sorted, key).clamp(max=len(self.key_sorted) - 1)
        ok = self.key_sorted[pos] == key
        return torch.where(ok, member.res_sorted[pos], 0.0), ok.float()

    def neighbour(self, member, g, r):
        s, n = torch.zeros(len(g), device=g.device), torch.zeros(len(g), device=g.device)
        for step in (self.gs["prv"], self.gs["nxt"]):
            h = g
            for _ in range(WINDOW):
                h = step[h]
                v, ok = self.lookup(member, h, r)
                s, n = s + v, n + ok
        win = s / (n + 1.0)
        nb, w = member.nbr[r], member.wts[r]
        v, ok = self.lookup(member, g[:, None].expand_as(nb), nb)
        ww = w * ok
        knn_w = ww.sum(1)
        knn = (ww * v).sum(1) / (knn_w + 0.05)
        nb, w = member.gnbr[g], member.gwts[g]
        v, ok = self.lookup(member, nb, r[:, None].expand_as(nb))
        ww = w * ok
        gknn_w = ww.sum(1)
        gknn = (ww * v).sum(1) / (gknn_w + 0.05)
        return win, knn, gknn, n, knn_w, gknn_w

    @torch.no_grad()
    def parts(self, member, g, r, off, neigh=True):
        keys = ("eta0", "inter", "log_r", "win", "knn", "gknn", "win_n", "knn_w", "gknn_w", "lvg", "lnr", "kind")
        out = {k: [] for k in keys}
        for i in range(0, len(g), CHUNK):
            gg, rr, oo = g[i:i + CHUNK].long(), r[i:i + CHUNK].long(), off[i:i + CHUNK]
            kind = (1 - self.rs["known"][rr]).long() + 2 * (1 - self.gs["known"][gg]).long()
            eta0, inter, log_r = member.net.cells(member.Gt, member.Rt, gg, rr, oo, kind)
            zero = torch.zeros_like(eta0)
            win = knn = gknn = win_n = knn_w = gknn_w = lvg = lnr = zero
            if neigh:
                win, knn, gknn, win_n, knn_w, gknn_w = self.neighbour(member, gg, rr)
                lvg, lnr = member.lvg[gg], member.lnr[rr]
            for k, x in zip(keys, (eta0, inter, log_r, win, knn, gknn, win_n, knn_w, gknn_w, lvg, lnr, kind)):
                out[k].append(x)
        P = {k: torch.cat(v) for k, v in out.items()}
        return {k: (v if k == "kind" else v.double()) for k, v in P.items()}


    def calibrate(self, member):
        self.residual_features(member)
        member.pseudo_pred = {}
        for name in KINDS:
            d = self.pseudo[name]
            src = "C" if name == "D" and len(d["y"]) < MIN_CALIB else name
            if len(self.pseudo[src]["y"]) < MIN_CALIB:
                member.theta[name] = torch.tensor(THETA0, dtype=torch.float64, device=self.device)
                log(f"calibration {name}: {len(d['y'])} pseudo cells, identity")
                continue
            ds = self.pseudo[src]
            P = self.parts(member, ds["g"], ds["r"], ds["off"])
            th, loss = fit_theta(P, ds["y"], FREE[src])
            raw = float(nb_nll_t(ds["y"], *calibrated(P, torch.tensor(THETA0, dtype=torch.float64,
                                                                       device=self.device))).mean())
            member.theta[name] = th
            log(f"calibration {name} (on pseudo {src}, {len(ds['y']):,} cells): raw {raw:.4f} -> {loss:.4f} | theta "
                f"{np.round(th.cpu().numpy(), 3).tolist()}")
            if len(d["y"]):
                Pd = P if src == name else self.parts(member, d["g"], d["r"], d["off"])
                eta, lr = calibrated(Pd, th)
                member.pseudo_pred[name] = (eta, lr)

    @staticmethod
    def mixture(preds):
        mu = torch.stack([torch.exp(e.clamp(-ETA_MAX, ETA_MAX)) for e, _ in preds])
        r = torch.stack([torch.exp(lr.clamp(LOG_R_MIN, LOG_R_MAX)) for _, lr in preds])
        m = mu.mean(0)
        second = (mu + mu ** 2 * (1.0 + 1.0 / r)).mean(0)
        excess = (second - m ** 2 - m).clamp(min=1e-12)
        return torch.log(m.clamp(min=1e-12)), torch.log(m ** 2 / excess)

    def fit_tau(self):
        for name in KINDS:
            preds = [m.pseudo_pred[name] for m in self.members if name in m.pseudo_pred]
            if len(preds) < len(self.members) or len(self.pseudo[name]["y"]) < MIN_CALIB:
                continue
            eta, lr = self.mixture(preds)
            y = self.pseudo[name]["y"]
            tau = torch.zeros(1, dtype=torch.float64, device=self.device, requires_grad=True)
            opt = torch.optim.LBFGS([tau], lr=1.0, max_iter=50, line_search_fn="strong_wolfe")

            def closure():
                opt.zero_grad()
                loss = nb_nll_t(y, eta, lr + tau).mean()
                loss.backward()
                return loss
            opt.step(closure)
            self.tau[name] = float(tau.detach()) if np.isfinite(float(tau.detach())) else 0.0
            with torch.no_grad():
                single = [float(nb_nll_t(y, e, l).mean()) for e, l in preds]
                mixed = float(nb_nll_t(y, eta, lr + self.tau[name]).mean())
            log(f"ensemble {name}: members {np.round(single, 4).tolist()} -> mixture {mixed:.4f} (tau {self.tau[name]:+.3f})")

    def meta_tables(self, gi, ri, g_code, r_code):
        gs, rs, G, R = self.gs, self.rs, len(self.genes), len(self.runs)
        n_g, n_r = np.bincount(gi, minlength=G), np.bincount(ri, minlength=R)
        n_genome = max(g_code.max(), r_code.max()) + 1
        k_runs = np.bincount(r_code[n_r > 0], minlength=n_genome)
        k_genes = np.bincount(g_code[n_g > 0], minlength=n_genome)
        unit = gs["unit_c_np"]
        prj = rs["prj_np"]
        prj_known = np.bincount(prj[n_r > 0], minlength=prj.max() + 1)[prj] - (n_r > 0)
        g_cols = [np.log1p(n_g), gs["has_hom_np"], np.log(np.bincount(unit)[unit]), np.log1p(k_genes[g_code]),
                  np.log1p(k_runs[g_code])]
        g_cols += list(gs["gm_level"].T)
        r_cols = [np.log1p(n_r), np.log1p(prj_known), rs["lsf"], np.log1p(k_runs[r_code])] + list(rs["sra"].T)
        dev = self.device
        self.g_meta = as_t(np.vstack([np.column_stack(g_cols), np.zeros((1, len(g_cols)))]), dev, torch.float32)
        self.r_meta = as_t(np.vstack([np.column_stack(r_cols), np.zeros((1, len(r_cols)))]), dev, torch.float32)

    @torch.no_grad()
    def stack_features(self, member, P0, preds, eta, lr, g, r, off):
        spread = torch.stack([e for e, _ in preds]).std(0) if len(preds) > 1 else torch.zeros_like(eta)
        lvl = eta - off
        cols = [lvl, lr, spread] + [P0[k] for k in ("inter", "log_r", "win", "knn", "gknn", "lvg", "lnr")]
        gs = self.gs
        if USE_OPERON:
            for side in ("prv", "nxt"):
                h = gs[side][g]
                v, ok = self.lookup(member, h, r)
                cols += [v * ok, ok, gs["same_" + side][g], gs["gap_" + side][g],
                         torch.where(self.gcnt[h] > 0, self.glvl[h] - lvl, 0.0), member.lvg[h]]
        if USE_XKNN:
            seen = self.xw[g] > 0
            cols += [torch.where(seen, self.xlvl[g] - lvl, 0.0), torch.where(seen, self.xlvl1[g] - lvl, 0.0),
                     self.xw[g], self.xtop[g]]
        X = torch.cat([torch.stack([c.float() for c in cols + [off]], 1), self.g_meta[g], self.r_meta[r]], 1)
        D = torch.cat([torch.stack([P0[k].float() for k in DESCRIPTORS], 1), self.g_meta[g][:, :5],
                       self.r_meta[r][:, :4]], 1)
        return X.cpu().numpy(), D.cpu().numpy()

    def stack_inputs(self):
        torch.cuda.empty_cache()
        self.stack_data = {}
        for name in KINDS:
            d = self.pseudo[name]
            preds = [m.pseudo_pred[name] for m in self.members if name in m.pseudo_pred]
            n = len(d["y"])
            if n < STACK_MIN or len(preds) < len(self.members):
                log(f"stacker {name}: {n:,} pseudo cells, none fitted")
                continue
            g, r = d["g"].long(), d["r"].long()
            eta, lr = self.mixture(preds) if len(preds) > 1 else preds[0]
            lr = lr + self.tau[name]
            P0 = self.parts(self.members[0], d["g"], d["r"], d["off"])
            X, D = self.stack_features(self.members[0], P0, preds, eta, lr, g, r, d["off"])
            grp = (self.rs["prj"][r] if name == "B" else self.gs["unit_c"][g]).cpu().numpy().astype(np.int64)
            h = (grp * 2654435761 + SPLIT_SEED) % 1000 / 1000.0
            split = (h < 0.4, (h >= 0.4) & (h < 0.7), h >= 0.7)
            if min(s.sum() for s in split) < STACK_MIN // 10:
                log(f"stacker {name}: splits {[int(s.sum()) for s in split]} too small, none fitted")
                continue
            self.stack_data[name] = dict(X=X, D=D, eta=eta.cpu(), lr=lr.cpu(), y=d["y"].cpu(), split=split)
            log(f"stacker {name}: {n:,} pseudo cells ({' / '.join(f'{s.sum():,}' for s in split)}), "
                f"{X.shape[1]} features")

    def fit_stacker(self, name, Dt):
        s = self.stack_data.pop(name)
        t0 = time.time()
        X, D, (s1, s2, s3) = s["X"], s["D"], s["split"]
        dev = self.device
        eta, lr, y = s["eta"].to(dev), s["lr"].to(dev), s["y"].to(dev)
        t = torch.log((y + 0.5) / (torch.exp(eta.clamp(-ETA_MAX, ETA_MAX)) + 0.5)).cpu().numpy()
        w = np.ones(len(X))
        msg = "weights none"
        if USE_SHIFT and len(Dt) >= SHIFT_MIN:
            w, auc, ess = domain_weights(D, Dt, SPLIT_SEED)
            msg = f"domain AUC {auc:.3f}, effective sample share {ess:.3f}"
        rows = [("pseudo", D.mean(0)), ("target", Dt.mean(0))]
        if USE_SHIFT:
            rows.insert(1, ("pseudo weighted", (D * w[:, None]).sum(0) / w.sum()))
        log(f"stacker {name} descriptors (mean): " + " | ".join(
            f"{k} " + " ".join(f"{x:.3f}" for x in v) for k, v in rows) + f" [{' '.join(DESC_NAMES)}]")
        m1 = boost(X[s1], t[s1], SPLIT_SEED, w[s1])
        d2 = m1.predict(X[s2])
        m2 = boost(np.column_stack([X[s2], d2]), np.log((t[s2] - d2) ** 2 + 0.01), SPLIT_SEED + 1, w[s2])
        d3 = m1.predict(X[s3])
        e3 = m2.predict(np.column_stack([X[s3], d3]))
        m3 = torch.as_tensor(s3, device=dev)
        th, loss, base = fit_stacked(eta[m3], lr[m3], torch.as_tensor(d3, dtype=torch.float64, device=dev),
                                     torch.as_tensor(e3, dtype=torch.float64, device=dev), y[m3],
                                     torch.as_tensor(w[s3], dtype=torch.float64, device=dev))
        keep = np.isfinite(loss) and loss < base
        if keep:
            self.stack[name] = (m1, m2, th)
        log(f"stacker {name}: {len(X):,} cells, {X.shape[1]} features, {msg}; held-out NLL mixture {base:.4f} -> "
            f"stacked {loss:.4f} ({'kept' if keep else 'dropped'}); theta {np.round(th.cpu().numpy(), 3).tolist()}; "
            f"{time.time() - t0:.0f}s")

    def absorb(self):
        d = self.absorb_cells
        self.gs["known"] = torch.maximum(self.gs["known"], self.absorb_g)
        self.rs["known"] = torch.maximum(self.rs["known"], self.absorb_r)
        n = len(d["y"])
        for member in self.members:
            net = member.net
            if n:
                with torch.no_grad():
                    tg = net.gene_tower(self.gs, self.gs["known"])
                    tr = net.run_tower(self.rs, self.rs["known"])
                opt = torch.optim.Adam([net.gfree, net.rfree], lr=3 * LR_FREE)
                gen = torch.Generator(device=self.device).manual_seed(SPLIT_SEED)
                for _ in range(ABSORB_STEPS):
                    idx = torch.randint(n, (min(BATCH, n),), device=self.device, generator=gen)
                    g, r = d["g"][idx], d["r"][idx]
                    Gt = tg + net.gfree * self.gs["known"][:, None]
                    Rt = tr + net.rfree * self.rs["known"][:, None]
                    eta0, inter, log_r = net.cells(Gt, Rt, g, r, d["off"][idx], torch.zeros_like(g))
                    loss = nb_nll_t(d["y"][idx], eta0 + inter, log_r).mean()
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    net.gfree.grad *= self.absorb_g[:, None]
                    net.rfree.grad *= self.absorb_r[:, None]
                    opt.step()
                log(f"absorbed {int(self.absorb_g.sum()):,} genes and {int(self.absorb_r.sum()):,} runs "
                    f"({n:,} cells), last NLL {loss.item():.4f}")
            self.finalise(member)


    def index(self, df):
        gc, gl = pair_index(df, "genome_id", "feature_id")
        rc, rl = pair_index(df, "genome_id", "run_accession")
        g = self.genes.get_indexer(gl)[gc]
        r = self.runs.get_indexer(rl)[rc]
        g = np.where(g >= 0, g, len(self.genes))
        r = np.where(r >= 0, r, len(self.runs))
        off = df["log_length_ratio"].to_numpy(dtype=np.float64) + df["log_size_factor"].to_numpy(dtype=np.float64)
        return g, r, off

    def predict(self, test_df, samples_df=None, genomes_dir=None):
        g, r, off = self.index(test_df)
        dev = self.device
        gt, rt, ot = as_t(g, dev), as_t(r, dev), as_t(off, dev, torch.float32)
        preds, kind, P0 = [], None, None
        for member in self.members:
            P = self.parts(member, gt, rt, ot)
            kind = P["kind"]
            P0 = P if P0 is None else P0
            eta = torch.empty_like(P["eta0"])
            lr = torch.empty_like(P["eta0"])
            for k, name in enumerate(KINDS):
                m = kind == k
                if m.any():
                    e, l = calibrated({x: v[m] for x, v in P.items() if x != "kind"}, member.theta[name])
                    eta[m], lr[m] = e, l
            preds.append((eta, lr))
        eta, lr = self.mixture(preds) if len(preds) > 1 else preds[0]
        for k, name in enumerate(KINDS):
            lr = torch.where(kind == k, lr + self.tau[name], lr)
        for k, name in enumerate(KINDS):
            m = kind == k
            if not m.any() or (name not in self.stack and name not in self.stack_data):
                continue
            sub = {x: v[m] for x, v in P0.items()}
            X, D = self.stack_features(self.members[0], sub, [(e[m], l[m]) for e, l in preds], eta[m], lr[m], gt[m],
                                       rt[m], ot[m])
            if name in self.stack_data:
                self.fit_stacker(name, D)
            if name not in self.stack:
                continue
            m1, m2, th = self.stack[name]
            delta = m1.predict(X)
            err = m2.predict(np.column_stack([X, delta]))
            eta[m], lr[m] = stacked(eta[m], lr[m], torch.as_tensor(delta, dtype=torch.float64, device=dev),
                                    torch.as_tensor(err, dtype=torch.float64, device=dev), th)
        mu = torch.exp(eta.clamp(-ETA_MAX, ETA_MAX)).cpu().numpy()
        sigma = torch.exp(lr.clamp(LOG_R_MIN, LOG_R_MAX)).cpu().numpy()
        kind = kind.cpu().numpy()
        log(f"predict rows {len(g):,}: kinds " + ", ".join(f"{n} {np.sum(kind == k):,}" for k, n in enumerate(KINDS))
            + f" | labels seen nowhere: gene {np.sum(g == len(self.genes))}, run {np.sum(r == len(self.runs))}")
        return np.column_stack([mu, sigma])


class Solver(GeneCompetSolver):

    name = "gene_expr_v5"

    def get_model(self, meta):
        return Model(meta["feature_annotations"], meta["submission_dir"])

    def fit(self, model, train_df, samples_df, genomes_dir):
        model.fit(train_df, samples_df, genomes_dir)
