import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

TARGETS = ["Y6", "Y12", "Y24", "Y48"]
FEATS = ["x_i", "y_i", "x_j", "y_j"]
BIG = np.int64(1 << 32)
WINDOWS = [20, 200]
UP = [1, 2, 3]
LINK_MIN_B = 100
LINK_MAX_GAP = 20
MAX_GROUP_SHARE = 0.01
PATH_DAS = [0, 1, -1, 2, 3]
PATH_MAX_GAP = 50
PATH_MIN_B = 10
PATH_STEPS = [1, 3, 10, 30]
CROSS_K = 20
CROSS_SLACK = 5
CROSS_Y_FRAC = 0.01
CROSS_X_FRAC = 0.02
SCAN_K = 60
HOLE_T = [0.002, 0.01, 0.05]
BUDGET_SPANS = [20, 60]
TE_K = 20
TE_DA = [-3, -2, -1, 0, 1, 2, 3]
PRIOR_M = 5.0

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


def find_csv(cols):
    for root in data_roots():
        for p in sorted(root.rglob("*.csv")):
            if list(pd.read_csv(p, nrows=0).columns) == cols:
                return p
    raise FileNotFoundError(cols)


def load(smoke=False):
    tr = pd.read_csv(find_csv(["id"] + FEATS + TARGETS))
    te = pd.read_csv(find_csv(["id"] + FEATS))
    assert len(tr) == 1370440 and len(te) == 362791, (len(tr), len(te))
    if smoke:
        tr, te = tr.sample(40000, random_state=0), te.sample(10000, random_state=0)
    allf = pd.concat([tr[["id"] + FEATS], te[["id"] + FEATS]], ignore_index=True)
    is_tr = np.r_[np.ones(len(tr), bool), np.zeros(len(te), bool)]
    labels = {t: np.r_[tr[t].values, np.zeros(len(te))].astype(np.float64) for t in TARGETS}
    say("train", tr.shape, "test", te.shape)
    return allf, is_tr, labels


class ChainIndex:
    def __init__(self, a, b, weights=None):
        self.a = a.astype(np.int64)
        self.b = b.astype(np.int64)
        key = self.a * BIG + self.b
        self.order = np.argsort(key, kind="stable")
        self.key = key[self.order]
        self.cs = {n: self.cumsum(w) for n, w in (weights or {}).items()}

    def cumsum(self, w):
        return np.concatenate([[0.0], np.cumsum(w[self.order].astype(np.float64))])

    def ranges(self, das, lo, hi, qa=None, qb=None):
        qa = self.a if qa is None else qa.astype(np.int64)
        qb = self.b if qb is None else qb.astype(np.int64)
        out = []
        for da in das:
            aa = qa + da
            i = np.searchsorted(self.key, aa * BIG + np.clip(qb + lo, 0, BIG - 1), "left")
            j = np.searchsorted(self.key, aa * BIG + np.clip(qb + hi, 0, BIG - 1), "right")
            out.append((i, j, (qb + hi >= 0) & (aa >= 0)))
        return out

    @staticmethod
    def range_sum(ranges, cs):
        return sum(np.where(ok, cs[j] - cs[i], 0.0) for i, j, ok in ranges)

    def sum(self, da, lo, hi, name):
        return self.range_sum(self.ranges([da], lo, hi), self.cs[name])

    def box(self, das, lo, hi, name):
        return self.range_sum(self.ranges(das, lo, hi), self.cs[name])

    def nearest(self, da, forward):
        aa = self.a + da
        n = len(self.key)
        if forward:
            i = np.minimum(np.searchsorted(self.key, aa * BIG + self.b + 1, "left"), n - 1)
            k = self.key[i]
            gap = k - aa * BIG - self.b
        else:
            i = np.maximum(np.searchsorted(self.key, aa * BIG + self.b - 1, "right") - 1, 0)
            k = self.key[i]
            gap = self.b - (k - aa * BIG)
        ok = (aa >= 0) & (k // BIG == aa) & (gap > 0)
        return np.where(ok, gap, np.nan), np.where(ok, self.order[i], -1)


def best_neighbour(idx, das, forward):
    gap, row = np.full(len(idx.b), np.nan), np.full(len(idx.b), -1)
    for da in das:
        g, r = idx.nearest(da, forward)
        better = np.nan_to_num(g, nan=np.inf) < np.nan_to_num(gap, nan=np.inf)
        gap, row = np.where(better, g, gap), np.where(better, r, row)
    return gap, row


def link_paths(idx):
    n = len(idx.b)
    back_das = [-d for d in PATH_DAS]
    ng, nr = best_neighbour(idx, PATH_DAS, True)
    pg, pr = best_neighbour(idx, back_das, False)
    r = np.arange(n)
    ok = (nr >= 0) & (ng <= PATH_MAX_GAP) & (idx.b >= PATH_MIN_B)
    s = np.where(ok, nr, 0)
    ok &= pr[s] == r
    fwd = idx.range_sum(idx.ranges(PATH_DAS, 1, np.nan_to_num(ng).astype(np.int64)), idx.cs["n"])
    back = idx.range_sum(idx.ranges(back_das, -np.nan_to_num(pg).astype(np.int64), -1), idx.cs["n"])
    ok &= (fwd == 1) & (back[s] == 1)
    return r[ok], nr[ok]


def path_features(prefix, a, b, src, dst):
    n = len(a)
    _, comp = connected_components(coo_matrix((np.ones(len(src)), (src, dst)), shape=(n, n)), directed=False)
    order = np.lexsort((b, comp))
    c, a_s, b_s = comp[order], a[order].astype(np.float64), b[order].astype(np.float64)
    first = np.r_[True, c[1:] != c[:-1]]
    starts = np.flatnonzero(first)
    grp = np.cumsum(first) - 1
    st = starts[grp]
    size = np.diff(np.r_[starts, n])[grp]
    en = st + size - 1
    i = np.arange(n)
    pos = i - st
    up = np.r_[False, (a_s[1:] > a_s[:-1]) & ~first[1:]]
    up_cnt = np.add.reduceat(up.astype(np.float64), starts)[grp]
    f = {"size": size.astype(np.float64), "pos": pos.astype(np.float64), "left": (size - 1 - pos).astype(np.float64),
         "pos_frac": pos / np.maximum(size - 1, 1), "net_a": a_s[en] - a_s[st], "span_b": b_s[en] - b_s[st],
         "up_rate": up_cnt / np.maximum(size - 1, 1)}
    f["slope"] = f["net_a"] / (f["span_b"] + 1)
    for k in PATH_STEPS:
        j, h = np.minimum(i + k, en), np.maximum(i - k, st)
        f[f"da_after{k}"] = a_s[j] - a_s
        f[f"db_after{k}"] = b_s[j] - b_s
        f[f"da_before{k}"] = a_s - a_s[h]
        f[f"db_before{k}"] = b_s - b_s[h]
    k = PATH_STEPS[-1]
    f["rate_after"] = f[f"da_after{k}"] / (f[f"db_after{k}"] + 1)
    f["rate_before"] = f[f"da_before{k}"] / (f[f"db_before{k}"] + 1)
    out = {}
    for name, v in f.items():
        u = np.empty(n)
        u[order] = v
        out[f"{prefix}p_{name}"] = u
    say(f"{prefix} paths: linked rows {len(src) / n:.3f}, rows in paths >= 2 {(size > 1).mean():.3f}, "
        f"rows in paths >= 10 {(size >= 10).mean():.3f}, longest {int(size.max())}")
    return out, comp


def step_counts(idx, qa, qb, das, lo, hi):
    return np.stack([idx.range_sum(idx.ranges([da], lo, hi, qa, qb), idx.cs["n"]) for da in das], axis=1)


def scan(idx, qa, qb, width):
    n = len(qa)
    found = np.zeros((n, SCAN_K + 1), bool)
    drift = np.full((n, SCAN_K + 1), np.nan, np.float32)
    for k in range(SCAN_K + 1):
        hi = width if k <= CROSS_K else width * k // CROSS_K
        (i, j, ok), = idx.ranges([k], -CROSS_SLACK, hi, qa, qb)
        hit = ok & (j > i)
        found[:, k] = hit
        drift[hit, k] = idx.key[i[hit]] % BIG - qb[hit]
    return found, drift


def first_true(m):
    return np.where(m.any(axis=1), m.argmax(axis=1), np.nan)


def last_true(m):
    return np.where(m.any(axis=1), m.shape[1] - 1 - m[:, ::-1].argmax(axis=1), np.nan)


def take(a, col):
    ok = ~np.isnan(col)
    out = np.full(len(a), np.nan)
    out[ok] = a[np.flatnonzero(ok), col[ok].astype(np.int64)]
    return out


def received(own, qa, qb, steps, top):
    s = np.nan_to_num(steps, nan=-1)
    t = np.nan_to_num(top, nan=0).astype(np.int64)
    tot = np.zeros(len(qa))
    for k in range(SCAN_K + 1):
        use = s >= k
        if not use.any():
            break
        (i, j, ok), = own.ranges([k], 1, t, qa, qb)
        tot += np.where(use & ok, np.maximum(own.cs["n"][j] - own.cs["n"][i], 0), 0)
    return np.where(np.isnan(steps) | np.isnan(top), np.nan, tot)


def account_holes(path, qa, found):
    lo = pd.Series(qa).groupby(path).transform("min").values
    hi = pd.Series(qa).groupby(path).transform("max").values
    r, k = np.nonzero(found)
    step = qa[r] + k
    keep = step < hi[r]
    uniq = np.unique(path[r[keep]] * BIG + step[keep])
    rec = np.bincount(uniq // BIG, minlength=path.max() + 1)[path].astype(np.float64)
    span = (hi - lo).astype(np.float64)
    holes = np.maximum(span - rec, 0)
    size = np.bincount(path)[path].astype(np.float64)
    return {"acc_holes": holes, "acc_hole_rate": np.where(span > 0, holes / np.maximum(span, 1), np.nan),
            "acc_holes_per_req": np.where(size > 1, holes / size, np.nan)}


def cross_role(names, other, own, qa, qb, width, path):
    v3, tp, bp = names
    found, drift = scan(other, qa, qb, width)
    after = found[:, :CROSS_K + 1]
    before = step_counts(other, qa, qb, range(-1, -CROSS_K - 1, -1), -width, CROSS_SLACK) > 0
    any_a, any_b = after.any(axis=1), before.any(axis=1)
    first = first_true(after)
    last = last_true(after)
    n_after = after.sum(axis=1).astype(np.float64)
    f = {"first_skip": first, "after_n": n_after, "after_holes": last + 1 - n_after,
         "prev_skip": np.where(any_b, before.argmax(axis=1), np.nan), "before_n": before.sum(axis=1).astype(np.float64)}
    f["after_hole_rate"] = f["after_holes"] / (last + 1)
    out = {f"{v3}_{k}": v for k, v in f.items()}

    cnt = np.cumsum(found, axis=1, dtype=np.int16)
    steps = np.arange(SCAN_K + 1)
    first60, last60 = first_true(found), last_true(found)
    holes60 = last60 + 1 - found.sum(axis=1)
    hole = ~found & (steps < np.nan_to_num(last60, nan=-1)[:, None])
    first_hole = first_true(hole)
    close = first_true(found & (steps > np.nan_to_num(first_hole, nan=SCAN_K + 1)[:, None]))
    dy_first, dy_close = take(drift, first60), take(drift, close)
    clock = qb.astype(np.float64) + 1
    t = {"first": first60, "n": found.sum(axis=1).astype(np.float64), "holes": holes60,
         "hole_rate": holes60 / (last60 + 1), "first_hole": first_hole, "dy_first": dy_first,
         "dy_first_rel": dy_first / clock, "dy_close": dy_close, "dy_close_rel": dy_close / clock}
    rel = drift / clock[:, None].astype(np.float32)
    for q, thr in enumerate(HOLE_T):
        lt = last_true(found & (rel <= thr))
        t[f"holes_t{q}"] = lt + 1 - take(cnt, lt)
    out.update({f"{tp}_{k}": v for k, v in t.items()})

    b = {"recv_first": received(own, qa, qb, first60, dy_first),
         "recv_close": received(own, qa, qb, close, dy_close)}
    b["close_share"] = (close + 1 - take(cnt, close)) / (b["recv_close"] + 1)
    for s in BUDGET_SPANS:
        ls = last_true(found[:, :s + 1])
        b[f"recv{s}"] = received(own, qa, qb, ls, take(drift, ls))
        b[f"budget{s}"] = (ls + 1 - take(cnt, ls)) / (b[f"recv{s}"] + 1)
    b.update(account_holes(path, qa, found))
    out.update({f"{bp}_{k}": v for k, v in b.items()})
    say(f"{v3} cross-role: rows with a later own record {any_a.mean():.3f} (within {SCAN_K}: {found.any(axis=1).mean():.3f}), "
        f"with an earlier one {any_b.mean():.3f}, first_skip==0 share among found {np.mean(first[any_a] == 0):.3f}, "
        f"rows with a closed hole {np.mean(~np.isnan(close)):.3f}, account holes > 0 {np.mean(b['acc_holes'] > 0):.3f}")
    return out


def cross_features(allf, paths):
    xi, yi, xj, yj = (allf[c].values.astype(np.int64) for c in FEATS)
    ones = {"n": np.ones(len(allf))}
    f = cross_role(("jr", "jt", "jb"), ChainIndex(xi, yi, ones), ChainIndex(xj, yj, ones), xj, yj,
                   (CROSS_SLACK + CROSS_Y_FRAC * yj).astype(np.int64), paths["dec"])
    f.update(cross_role(("id", "it", "ib"), ChainIndex(yj, xj, ones), ChainIndex(yi, xi, ones), yi, xi,
                        (CROSS_SLACK + CROSS_X_FRAC * xi).astype(np.int64), paths["req"]))
    return f


def chain_side(prefix, a, b, n):
    idx = ChainIndex(a, b, {"n": np.ones(n)})
    f = {f"{prefix}_exact": idx.sum(0, 0, 0, "n") - 1}
    for k in WINDOWS:
        f[f"{prefix}_same_after{k}"] = idx.sum(0, 1, k, "n")
        f[f"{prefix}_same_before{k}"] = idx.sum(0, -k, -1, "n")
        f[f"{prefix}_up_after{k}"] = idx.box(UP, 1, k, "n")
        f[f"{prefix}_down_before{k}"] = idx.box([-d for d in UP], -k, -1, "n")
        f[f"{prefix}_up_frac{k}"] = f[f"{prefix}_up_after{k}"] / np.maximum(
            f[f"{prefix}_up_after{k}"] + f[f"{prefix}_same_after{k}"], 1)
        f[f"{prefix}_down_frac{k}"] = f[f"{prefix}_down_before{k}"] / np.maximum(
            f[f"{prefix}_down_before{k}"] + f[f"{prefix}_same_before{k}"], 1)

    nxt_same, row_same = idx.nearest(0, True)
    ups = [idx.nearest(d, True) for d in UP]
    nxt_up = np.fmin.reduce([g for g, _ in ups])
    f[f"{prefix}_next_gap_same"] = nxt_same
    f[f"{prefix}_next_gap_up"] = nxt_up
    f[f"{prefix}_next_is_up"] = np.where(np.isnan(nxt_same) & np.isnan(nxt_up), np.nan,
                                         (np.nan_to_num(nxt_up, nan=np.inf) < np.nan_to_num(nxt_same, nan=np.inf))
                                         .astype(float))
    f[f"{prefix}_prev_gap_same"] = idx.nearest(0, False)[0]
    f[f"{prefix}_prev_gap_down"] = np.fmin.reduce([idx.nearest(-d, False)[0] for d in UP])

    best_gap, best_row = nxt_same.copy(), row_same.copy()
    for g, r in ups:
        better = np.nan_to_num(g, nan=np.inf) < np.nan_to_num(best_gap, nan=np.inf)
        best_gap = np.where(better, g, best_gap)
        best_row = np.where(better, r, best_row)
    link = (best_row >= 0) & (best_gap <= LINK_MAX_GAP) & (idx.b >= LINK_MIN_B)
    src = np.r_[np.flatnonzero(link), idx.order[:-1]]
    dst = np.r_[best_row[link], idx.order[1:]]
    dup = np.r_[np.ones(link.sum(), bool), idx.key[1:] == idx.key[:-1]]
    g = coo_matrix((np.ones(dup.sum()), (src[dup], dst[dup])), shape=(n, n))
    _, comp = connected_components(g, directed=False)
    comp = pd.Series(comp)
    size = comp.map(comp.value_counts()).values
    big = size > MAX_GROUP_SHARE * n
    pair_id = pd.factorize(idx.a * BIG + idx.b)[0]
    comp = np.where(big, comp.max() + 1 + pair_id, comp.values)
    say(f"{prefix} components: linked rows {link.mean():.3f}, components {len(np.unique(comp))}, "
        f"rows split back into pairs (component > {MAX_GROUP_SHARE:.0%}) {big.mean():.3f}")

    d = pd.DataFrame({"c": comp, "a": idx.a, "b": idx.b})
    grp = d.groupby("c")
    size = grp["c"].transform("size").values
    aspan = (grp["a"].transform("max") - grp["a"].transform("min")).values
    bmin = grp["b"].transform("min").values
    bspan = grp["b"].transform("max").values - bmin
    f[f"{prefix}_comp_size"] = size
    f[f"{prefix}_comp_aspan"] = aspan
    f[f"{prefix}_comp_bspan"] = bspan
    f[f"{prefix}_comp_slope"] = aspan / (bspan + 1)
    f[f"{prefix}_comp_pos"] = (idx.b - bmin) / (bspan + 1)
    return f, comp


def base_features(allf):
    lx = {c: np.log1p(allf[c].values.astype(np.float64)) for c in FEATS}
    r_i, r_j = lx["x_i"] - lx["y_i"], lx["x_j"] - lx["y_j"]
    f = {c: allf[c].values.astype(np.float64) for c in FEATS}
    f.update({"r_i": r_i, "r_j": r_j, "rj_minus_ri": r_j - r_i, "lxj_plus_lyi": lx["x_j"] + lx["y_i"],
              "lyi_minus_lyj": lx["y_i"] - lx["y_j"], "lxi_minus_lxj": lx["x_i"] - lx["x_j"],
              "lyi_minus_lxj": lx["y_i"] - lx["x_j"], "lxi_minus_lyj": lx["x_i"] - lx["y_j"]})
    for c in ["x_i", "y_j"]:
        f[f"cnt_{c}"] = allf[c].map(allf[c].value_counts()).values.astype(np.float64)
    return f


def requester_te(allf, is_tr, labels, fold, req_path):
    idx = ChainIndex(allf.y_i.values, allf.x_i.values)
    boxes = {"exact": idx.ranges([0], 0, 0), "near": idx.ranges(TE_DA, -TE_K, TE_K)}
    out = {f"te_req_{s}_{t}": np.zeros(len(allf)) for s in list(boxes) + ["path"] for t in TARGETS + ["n"]}
    for k in sorted(set(fold[is_tr])) + [None]:
        use = is_tr & (fold != k) if k is not None else is_tr
        rows = (fold == k) if k is not None else ~is_tr
        cs = {"n": idx.cumsum(use.astype(float))}
        cs.update({t: idx.cumsum(labels[t] * use) for t in TARGETS})
        for s, rg in boxes.items():
            cnt = idx.range_sum(rg, cs["n"])
            out[f"te_req_{s}_n"][rows] = cnt[rows]
            for t in TARGETS:
                prior = labels[t][use].mean()
                out[f"te_req_{s}_{t}"][rows] = ((idx.range_sum(rg, cs[t]) + PRIOR_M * prior) / (cnt + PRIOR_M))[rows]
        cnt = np.bincount(req_path, weights=use.astype(float))[req_path]
        out["te_req_path_n"][rows] = cnt[rows]
        for t in TARGETS:
            prior = labels[t][use].mean()
            hits = np.bincount(req_path, weights=labels[t] * use)[req_path]
            out[f"te_req_path_{t}"][rows] = ((hits + PRIOR_M * prior) / (cnt + PRIOR_M))[rows]
    return out


def build_features(allf):
    n = len(allf)
    f = base_features(allf)
    say("base features", len(f))
    req, _ = chain_side("req", allf.y_i.values, allf.x_i.values, n)
    dec, dec_comp = chain_side("dec", allf.x_j.values, allf.y_j.values, n)
    f.update(req)
    f.update(dec)
    say("V1 features without TE", len(f))
    paths = {}
    for prefix, a, b in [("req", allf.y_i.values, allf.x_i.values), ("dec", allf.x_j.values, allf.y_j.values)]:
        idx = ChainIndex(a, b, {"n": np.ones(n)})
        pf, paths[prefix] = path_features(prefix, a, b, *link_paths(idx))
        f.update(pf)
    f.update(cross_features(allf, paths))
    say("features without TE", len(f))
    return pd.DataFrame(f), dec_comp, paths
