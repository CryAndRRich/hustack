import numpy as np
import pandas as pd

import config
from runtime import log, step


def load_mapping(path):
    m = pd.read_csv(path, usecols=["app_id", "category"]).dropna()
    m["app_id"] = pd.to_numeric(m["app_id"], errors="coerce")
    m = m.dropna(subset=["app_id"])
    dup = int(m["app_id"].duplicated().sum())
    m = m.drop_duplicates("app_id")
    cats = sorted(m["category"].astype(str).unique())
    code = pd.Series(m["category"].astype(str).map({c: i for i, c in enumerate(cats)}).values,
                     index=m["app_id"].astype(np.int64).values)
    log("mapping: %d apps, %d categories, %d duplicate app_id rows dropped" % (len(code), len(cats), dup))
    return cats, code


def feature_names(cats):
    return [config.CAT_PREFIX + "n_apps", config.CAT_PREFIX + "n_mapped"] + [config.CAT_PREFIX + c for c in cats]


def _features(path, dia, code, n_cat, with_date):
    keys, mats = [], []
    usecols = ["hashed_aaid", "app_ids"] + (["date"] if with_date else [])
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=config.CHUNK, dtype=str, **dia):
        chunk = chunk.reset_index(drop=True)
        s = chunk["app_ids"].fillna("").str.replace(r'[\[\]\s"]', "", regex=True)
        lists = s.str.split(",")
        n_apps = np.where(s.values == "", 0, lists.str.len().values).astype(np.float32)
        flat = pd.to_numeric(lists.explode(), errors="coerce")
        flat = flat[flat.notna()].astype(np.int64)
        cat = flat.map(code)
        ok = cat.notna().values
        counts = np.zeros((len(chunk), n_cat), dtype=np.float32)
        np.add.at(counts, (flat.index.values[ok], cat.values[ok].astype(np.int64)), 1)
        mats.append(np.column_stack([n_apps, counts.sum(axis=1), counts]))
        k = chunk[["hashed_aaid"]].copy()
        if with_date:
            k["date"] = pd.to_datetime(chunk["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        keys.append(k)
        step("catalog %s: %d rows" % ("training" if with_date else "candidates", sum(len(x) for x in keys)))
    return pd.concat(keys, ignore_index=True), np.vstack(mats).astype(np.float32)


def attach(left_keys, path, dia, code, n_cat, with_date):
    keys, mat = _features(path, dia, code, n_cat, with_date)
    on = ["hashed_aaid", "date"] if with_date else ["hashed_aaid"]
    keys["crow"] = np.arange(len(keys))
    dup = int(keys.duplicated(on).sum())
    keys = keys.drop_duplicates(on)
    left = left_keys[on].copy()
    left["row"] = np.arange(len(left))
    m = left.merge(keys, on=on, how="inner")
    out = np.full((len(left), mat.shape[1]), np.nan, dtype=np.float32)
    out[m["row"].values] = mat[m["crow"].values]
    log("catalog join on %s: %d of %d rows matched (%.4f); %d duplicate catalog keys dropped"
        % (on, len(m), len(left), len(m) / max(len(left), 1), dup))
    return out
