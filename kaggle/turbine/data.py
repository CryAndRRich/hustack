import json
from collections import Counter

import numpy as np
import pandas as pd

import config
from runtime import log, step

NON_FEATURES = {"hashed_aaid", "install", "events", "install_timestamp", "d", "ad_revenue"}
EVENT_KEYS = frozenset({"timestamp", "value_usd", "name"})


def revenue_of(cell):
    if not isinstance(cell, str) or cell in ("", "[]"):
        return 0.0
    total = 0.0
    for item in json.loads(cell):
        if frozenset(item) != EVENT_KEYS:
            raise ValueError("event keys %s" % sorted(item))
        if item["name"] == "ad_revenue":
            total += float(item["value_usd"])
    return total


def family_of(col, cats):
    for cat in cats:
        token = "_" + cat + "_"
        if token in col:
            return col.replace(token, "_", 1)
        if col.endswith("_" + cat):
            return col[: -len(cat) - 1]
    return None


def families(feats, cats):
    groups = {}
    longest_first = sorted(cats, key=len, reverse=True)
    for i, c in enumerate(feats):
        f = family_of(c, longest_first)
        if f is not None:
            groups.setdefault(f, []).append(i)
    return [("sum_" + f, np.array(ix)) for f, ix in groups.items() if len(ix) >= 2]


def numeric_block(frame, feats, coerced):
    sub = frame[feats]
    for c in sub.columns[sub.dtypes == object]:
        num = pd.to_numeric(sub[c], errors="coerce")
        coerced[c] += int((sub[c].notna() & num.isna()).sum())
        sub = sub.assign(**{c: num})
    return sub.to_numpy(dtype=np.float32)


def family_block(block, fams):
    out = np.empty((len(block), len(fams)), dtype=np.float32)
    for j, (_f, ix) in enumerate(fams):
        part = block[:, ix]
        present = (~np.isnan(part)).any(axis=1)
        out[:, j] = np.where(present, np.nansum(part, axis=1), np.nan)
    return out


def load_training(path, dia, cats, smoke):
    cols = pd.read_csv(path, nrows=0, **dia).columns
    feats = [c for c in cols if c not in NON_FEATURES]
    fams = families(feats, cats)
    names = feats + [f for f, _ in fams]
    cap = config.SMOKE_TRAIN_ROWS if smoke else config.TRAIN_ROW_CAP
    X = np.empty((cap, len(names)), dtype=np.float32)
    rng = np.random.default_rng(config.SEED)
    ids, inst, rev, d = [], [], [], []
    coerced = Counter()
    pos = scanned = 0
    reader = pd.read_csv(path, usecols=["hashed_aaid", "install", "events", "d"] + feats,
                         chunksize=config.CHUNK, low_memory=False,
                         nrows=config.SMOKE_TRAIN_ROWS if smoke else None, **dia)
    for chunk in reader:
        scanned += len(chunk)
        ins = pd.to_numeric(chunk["install"], errors="coerce").fillna(0).astype(np.int8).values
        keep = (ins == 1) | (rng.random(len(chunk)) < config.NONINSTALL_KEEP)
        sub = chunk.loc[keep]
        n = len(sub)
        if pos + n > cap:
            raise RuntimeError("training rows exceed the preallocated cap %d" % cap)
        block = numeric_block(sub, feats, coerced)
        X[pos:pos + n, :len(feats)] = block
        X[pos:pos + n, len(feats):] = family_block(block, fams)
        ids.append(sub["hashed_aaid"].astype(str).values)
        inst.append(ins[keep])
        rev.append(np.array([revenue_of(c) for c in sub["events"].values]))
        d.append(sub["d"].astype(str).values)
        pos += n
        step("training: %d rows scanned, %d kept" % (scanned, pos))
    X = X[:pos]
    inst = np.concatenate(inst).astype(np.int8)
    out = {"ids": np.concatenate(ids), "inst": inst, "rev": np.concatenate(rev), "d": np.concatenate(d),
           "sw": np.where(inst == 1, 1.0, 1.0 / config.NONINSTALL_KEEP)}
    bad = {c: n for c, n in coerced.items() if n}
    log("non-numeric values coerced to NaN: %d columns %s" % (len(bad), list(bad.items())[:5]))

    present = np.zeros(len(names), dtype=np.int64)
    for j in range(0, len(names), 200):
        present[j:j + 200] = (~np.isnan(X[:, j:j + 200])).sum(axis=0)
    keep_cols = present >= (config.MIN_PRESENT if not smoke else 5)
    out["X"] = np.ascontiguousarray(X[:, keep_cols])
    del X
    out["names"] = [n for n, k in zip(names, keep_cols) if k]
    out["spec"] = {"feats": feats, "fams": fams, "keep_cols": keep_cols}
    log("training kept rows=%d (installs %d, payers %d, revenue %.2f) of %d scanned"
        % (pos, int(inst.sum()), int((out["rev"] > 0).sum()), out["rev"].sum(), scanned))
    log("features: %d raw + %d families -> %d kept (>= %d present)"
        % (len(feats), len(fams), int(keep_cols.sum()), config.MIN_PRESENT))
    step("training matrix ready")
    return out


def load_candidates(path, dia, spec, smoke):
    feats, fams, keep_cols = spec["feats"], spec["fams"], spec["keep_cols"]
    cap = config.SMOKE_CAND_ROWS if smoke else config.CAND_ROW_CAP
    X = np.empty((cap, int(keep_cols.sum())), dtype=np.float32)
    ids, adrev = [], []
    coerced = Counter()
    pos = 0
    reader = pd.read_csv(path, usecols=["hashed_aaid", "ad_revenue"] + feats, chunksize=config.CHUNK,
                         low_memory=False, nrows=config.SMOKE_CAND_ROWS if smoke else None, **dia)
    for chunk in reader:
        n = len(chunk)
        if pos + n > cap:
            raise RuntimeError("candidate rows exceed the preallocated cap %d" % cap)
        block = numeric_block(chunk, feats, coerced)
        full = np.hstack([block, family_block(block, fams)])
        X[pos:pos + n] = full[:, keep_cols]
        ids.append(chunk["hashed_aaid"].astype(str).values)
        adrev.append(pd.to_numeric(chunk["ad_revenue"], errors="coerce").fillna(0).values)
        pos += n
        step("candidates: %d rows" % pos)
    bad = {c: n for c, n in coerced.items() if n}
    log("candidates: non-numeric values coerced to NaN: %d columns %s" % (len(bad), list(bad.items())[:5]))
    out = {"X": X[:pos], "ids": np.concatenate(ids), "ad_revenue": np.concatenate(adrev)}
    if len(set(out["ids"])) != pos:
        raise ValueError("candidate ids are not unique")
    log("candidates: %d rows, %d with ad_revenue" % (pos, int((out["ad_revenue"] > 0).sum())))
    return out
