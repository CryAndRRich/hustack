import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from common.runtime import WORK, as_text, find, log, run_workers

TMP = Path("/tmp/gm")
WORKER = Path("/tmp/gene_expr/gene_model.py")
VARIANTS = ("esm", "prom")
FOLDS, FOLD_SEED = 5, 7
OUT_ZIP = WORK / "gene-expr-genemodel.zip"


def zscore(E, rows, chunk=100_000):
    E32 = E.astype(np.float32)
    mu, sd = E32.mean(0), E32.std(0) + 1e-6
    out = np.zeros((len(rows), E.shape[1]), dtype=np.float16)
    for i in range(0, len(rows), chunk):
        r = rows[i:i + chunk]
        ok = r >= 0
        out[i:i + chunk][ok] = ((E32[r[ok]] - mu) / sd).astype(np.float16)
    return out


def positions(df, a, b, index):
    ca, cb = (df[c].astype("category") for c in (a, b))
    la, lb = ca.cat.categories.astype(str), cb.cat.categories.astype(str)
    code = ca.cat.codes.to_numpy().astype(np.int64) * len(lb) + cb.cat.codes.to_numpy()
    uniq, inv = np.unique(code, return_inverse=True)
    labels = la[uniq // len(lb)] + "|" + lb[uniq % len(lb)]
    return index.get_indexer(labels)[inv]


def units(F, esm_rows):
    G = len(F)
    hom = pd.factorize((as_text(F["ref_genome_id"]) + "|" + as_text(F["ref_feature_id"]))
                       .where(F["ref_feature_id"].notna()))[0]
    H = int(hom.max()) + 1
    src = np.concatenate([np.flatnonzero(hom >= 0), np.flatnonzero(esm_rows >= 0)])
    dst = np.concatenate([G + hom[hom >= 0], G + H + esm_rows[esm_rows >= 0]])
    n = G + H + int(esm_rows.max()) + 1
    _, lab = connected_components(coo_matrix((np.ones(len(src)), (src, dst)), shape=(n, n)), directed=False)
    lab = pd.factorize(lab[:G])[0]
    log(f"units: {lab.max() + 1:,} over {G:,} genes (homolog keys {np.mean(hom >= 0):.3f}, with a protein "
        f"{np.mean(esm_rows >= 0):.3f})")
    return lab


def prepare():
    TMP.mkdir(parents=True, exist_ok=True)
    feat = pd.read_csv(find("feature_annotation.csv"), low_memory=False)
    cond = pd.read_csv(find("condition_annotation.csv"), low_memory=False)
    F = feat.drop_duplicates(["genome_id", "feature_id"]).reset_index(drop=True)
    genes = pd.Index(as_text(F["genome_id"]) + "|" + as_text(F["feature_id"]))
    keys = pd.read_parquet(find("gene_seq_keys.parquet"))
    pos = genes.get_indexer(as_text(keys["genome_id"]) + "|" + as_text(keys["feature_id"]))
    assert (pos >= 0).all() and len(np.unique(pos)) == len(genes) == len(keys), "gene_seq_keys does not match"
    K = keys.iloc[np.argsort(pos)].reset_index(drop=True)

    esm_rows = K["esm_row"].to_numpy()
    length = pd.to_numeric(F["Length"], errors="coerce").to_numpy(dtype=float)
    loglen = np.log(np.clip(np.nan_to_num(length, nan=np.nanmedian(length)), 1.0, None))
    basic = np.column_stack([(esm_rows >= 0), (loglen - loglen.mean()) / loglen.std()]).astype(np.float16)
    np.save(TMP / "X_esm.npy", np.hstack([zscore(np.load(find("esm_full.npy"), mmap_mode="r"), esm_rows), basic]))
    pr = K["prom_row"].to_numpy()
    plen = np.log(np.clip(K["prom_len"].to_numpy(dtype=float), 1.0, None))
    flags = np.column_stack([pr >= 0, K["prom_own"].to_numpy(dtype=bool), np.where(pr >= 0, (plen - 5.0), 0.0),
                             np.clip(K["up_gap"].to_numpy(dtype=float), -512, 512) / 256.0]).astype(np.float16)
    c3p = [zscore(np.load(find(f"{n}.npy"), mmap_mode="r"), pr) for n in ("c3p_5m", "c3p_100m")]
    np.save(TMP / "X_prom.npy", np.hstack(c3p + [flags]))
    log(f"features: esm {1280 + basic.shape[1]}, prom extra {sum(c.shape[1] for c in c3p) + flags.shape[1]}; "
        f"protein {np.mean(esm_rows >= 0):.4f}, promoter {np.mean(pr >= 0):.4f}")

    runs = pd.Index(as_text(cond["genome_id"]) + "|" + as_text(cond["run_accession"]))
    obs = pd.read_parquet(find("read_count_train.parquet"),
                          columns=["genome_id", "feature_id", "run_accession", "readcount", "status"])
    g = positions(obs, "genome_id", "feature_id", genes)
    r = positions(obs, "genome_id", "run_accession", runs)
    status = obs["status"].astype("category")
    st = {k: (status == k).to_numpy() for k in ("Train", "eval_C")}
    y = obs["readcount"].to_numpy(dtype=np.float32)
    del obs, status
    assert (g >= 0).all() and (r >= 0).all(), "a counted gene or run is missing from the annotation"
    off = (F["log_length_ratio"].to_numpy(dtype=np.float32)[g]
           + cond["log_size_factor"].to_numpy(dtype=np.float32)[r])
    for name, m in (("train", st["Train"]), ("evalc", st["eval_C"])):
        for k, v in (("g", g[m].astype(np.int32)), ("r", r[m].astype(np.int32)), ("y", y[m]), ("off", off[m])):
            np.save(TMP / f"{name}_{k}.npy", v)
    np.save(TMP / "n_runs.npy", np.array(len(runs)))
    np.save(TMP / "gene_genome.npy", pd.factorize(as_text(F["genome_id"]))[0].astype(np.int32))

    has_train = np.bincount(g[st["Train"]], minlength=len(genes)) > 0
    lab = units(F, esm_rows)
    perm = np.random.default_rng(FOLD_SEED).permutation(lab.max() + 1)
    fold = np.where(has_train, perm[lab] % FOLDS, -1).astype(np.int32)
    np.save(TMP / "gene_fold.npy", fold)
    eval_c = np.zeros(len(genes), dtype=bool)
    eval_c[g[st["eval_C"]]] = True
    assert not (eval_c & has_train).any(), "an eval C gene has Train cells"
    shared = np.isin(lab[eval_c], np.unique(lab[has_train]))
    log(f"eval C genes {eval_c.sum():,}: {shared.mean():.3f} sit in a unit that holds a Train gene (0 = the units "
        f"reproduce the C design)")
    log(f"cells: Train {int(st['Train'].sum()):,}, eval C {int(st['eval_C'].sum()):,}; genes with Train "
        f"cells {has_train.sum():,}; fold sizes {np.bincount(fold[fold >= 0]).tolist()}")
    out = F[["genome_id", "feature_id"]].astype(str).copy()
    out["fold"] = fold
    out["unit"] = lab.astype(np.int32)
    return out


def run_all(smoke):
    run_workers([([sys.executable, WORKER, "--data", TMP, "--variant", v, "--out", TMP / f"gene_model_{v}.npy"]
                  + (["--smoke"] if smoke else []), i % 2, f"[{v}{' smoke' if smoke else ''}]")
                 for i, v in enumerate(VARIANTS)])


def main():
    keys = prepare()
    run_all(smoke=True)
    log("smoke pass done")
    run_all(smoke=False)
    keys.to_parquet(TMP / "gene_model_keys.parquet", index=False)
    names = [f"gene_model_{v}.npy" for v in VARIANTS] + ["gene_model_keys.parquet"]
    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_STORED) as z:
        for n in names:
            z.write(TMP / n, n)
    shutil.rmtree(TMP, ignore_errors=True)
    shutil.rmtree(WORKER.parent, ignore_errors=True)
    log(f"wrote {OUT_ZIP} ({OUT_ZIP.stat().st_size / 1e6:.0f} MB); /kaggle/working holds "
        f"{sorted(p.name for p in WORK.iterdir())}")


main()
