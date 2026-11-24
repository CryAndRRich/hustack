import json
import multiprocessing as mp
import os
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common.runtime import WORK, log, pip_install, walk
from gene_expr.sequences import WORKER, esm_unique

TMP = Path("/tmp/gene_emb")
TEXT_MODEL = "BAAI/bge-small-en-v1.5"
ESM_DIM, TEXT_DIM = 64, 32
TEXT_MAX_TOKENS = 256
OUT_ZIP = WORK / "gene-expr-emb.zip"


def parse_proteins(path):
    from Bio import SeqIO
    rows = []
    for rec in SeqIO.parse(str(path), "genbank"):
        for f in rec.features:
            if f.type == "CDS" and f.location is not None:
                rows.append((path.stem, rec.id, rec.name, int(f.location.start) + 1,
                             "-" if f.location.strand == -1 else "+", f.qualifiers.get("translation", [""])[0]))
    return pd.DataFrame(rows, columns=["genome_id", "chr_id", "chr_name", "start", "strand", "seq"])


def gene_proteins(feat):
    F = feat.drop_duplicates(["genome_id", "feature_id"]).reset_index(drop=True)
    F[["genome_id", "feature_id"]] = F[["genome_id", "feature_id"]].astype(str)
    F["chr"] = F["Chr"].astype(str).str.split(";").str[0]
    F["start"] = pd.to_numeric(F["Start"].astype(str).str.split(";").str[0], errors="coerce")
    F["strand"] = F["Strand"].astype(str).str.split(";").str[0]
    files = {p.stem: p for p in walk(".gbff")}
    paths = [files[g] for g in sorted(F["genome_id"].unique()) if g in files]
    with mp.get_context("fork").Pool(os.cpu_count()) as pool:
        gb = pd.concat(pool.map(parse_proteins, paths, chunksize=1), ignore_index=True)
    gb["start"] = gb["start"].astype(float)
    pos = ["genome_id", "chr", "start", "strand"]
    got = F[pos].merge(gb.rename(columns={"chr_id": "chr"}).drop(columns="chr_name").drop_duplicates(pos),
                       on=pos, how="left")["seq"]
    miss = got.isna().to_numpy()
    alt = F.loc[miss, pos].merge(gb.rename(columns={"chr_name": "chr"}).drop(columns="chr_id").drop_duplicates(pos),
                                 on=pos, how="left")["seq"]
    got = got.to_numpy(dtype=object)
    got[miss] = alt.to_numpy(dtype=object)
    F["seq"] = pd.Series(got).fillna("").astype(str).str.rstrip("*")
    log(f"genes {len(F):,}: CDS matched {np.mean(~pd.isna(got)):.4f}, with a protein {np.mean(F['seq'] != ''):.4f}, "
        f"length {np.quantile(F['seq'].str.len(), [0, .5, .99, 1]).round().tolist()}")
    return F[["genome_id", "feature_id", "seq"]]


def esm_embeddings(genes):
    E, codes = esm_unique(genes["seq"], TMP)
    P = pca(E, ESM_DIM, "ESM2")
    out = np.zeros((len(genes), ESM_DIM), dtype=np.float16)
    out[codes >= 0] = P[codes[codes >= 0]].astype(np.float16)
    keys = genes[["genome_id", "feature_id"]].copy()
    keys["has_esm"] = codes >= 0
    return out, keys


def pca(X, k, name):
    mu = X.mean(0)
    Z = X - mu
    C = (Z.T @ Z).astype(np.float64) / len(X)
    w, V = np.linalg.eigh(C)
    w, V = w[::-1], V[:, ::-1]
    log(f"{name} PCA: {k} of {X.shape[1]} dims keep {w[:k].sum() / w.sum():.3f} of the variance")
    return (Z @ V[:, :k].astype(np.float32)).astype(np.float32)


def run_texts(cond):
    docs = [json.loads(p.read_text()) for p in walk("_curated.json")]
    if not docs:
        for z in walk("experiment.zip"):
            with zipfile.ZipFile(z) as zf:
                docs += [json.loads(zf.read(n)) for n in zf.namelist() if n.endswith("_curated.json")]
    text = {}
    for d in docs:
        for e in d.get("experiments") or []:
            st = e.get("strain") or {}
            pert = e.get("perturbation") or {}
            parts = [e.get("raw_sample_title"), e.get("condition_summary"), st.get("background_strain"),
                     "; ".join(m.get("raw_genotype_string") or "" for m in st.get("genetic_modifications") or []),
                     pert.get("stressor_or_compound"), pert.get("dose_or_intensity"), pert.get("exposure_duration")]
            text[e.get("experiment_accession")] = " | ".join(str(p) for p in parts if p not in (None, "", []))
    R = cond.drop_duplicates(["genome_id", "run_accession"]).reset_index(drop=True)
    fallback = R.apply(lambda row: " | ".join(f"{c}: {row[c]}" for c in (
        "species", "medium_name", "medium_category", "growth_phase", "temperature_celsius", "oxygen_condition",
        "perturbation_category", "perturbation_compound", "is_wild_type") if pd.notna(row[c])), axis=1)
    has = R["experiment_accession"].map(lambda x: x in text).to_numpy(dtype=bool)
    texts = np.where(has, R["experiment_accession"].map(text).fillna(""), fallback)
    log(f"runs {len(R):,}: experiment text {has.mean():.3f}; example: {texts[0][:200]!r}")
    keys = R[["genome_id", "run_accession"]].astype(str).copy()
    keys["has_exp"] = has
    return list(texts), keys


def text_embeddings(texts):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TEXT_MODEL)
    model = AutoModel.from_pretrained(TEXT_MODEL).cuda().eval()
    out = []
    with torch.inference_mode():
        for i in range(0, len(texts), 256):
            enc = tok(texts[i:i + 256], padding=True, truncation=True, max_length=TEXT_MAX_TOKENS, return_tensors="pt")
            h = model(**{k: v.cuda() for k, v in enc.items()}).last_hidden_state[:, 0]
            out.append(torch.nn.functional.normalize(h, dim=1).float().cpu().numpy())
    return pca(np.vstack(out), TEXT_DIM, "text").astype(np.float16)


def main():
    pip_install("Bio", "biopython")
    feat = pd.read_csv(next(walk("feature_annotation.csv")), low_memory=False)
    cond = pd.read_csv(next(walk("condition_annotation.csv")), low_memory=False)
    genes = gene_proteins(feat)
    esm, gkeys = esm_embeddings(genes)
    texts, rkeys = run_texts(cond)
    txt = text_embeddings(texts)
    assert np.isfinite(esm.astype(np.float32)).all() and np.isfinite(txt.astype(np.float32)).all()
    TMP.mkdir(parents=True, exist_ok=True)
    np.save(TMP / "gene_esm.npy", esm)
    gkeys.to_parquet(TMP / "gene_keys.parquet", index=False)
    np.save(TMP / "run_text.npy", txt)
    rkeys.to_parquet(TMP / "run_keys.parquet", index=False)
    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_STORED) as z:
        for name in ("gene_esm.npy", "gene_keys.parquet", "run_text.npy", "run_keys.parquet"):
            z.write(TMP / name, name)
    shutil.rmtree(TMP, ignore_errors=True)
    shutil.rmtree(WORKER.parent, ignore_errors=True)
    log(f"wrote {OUT_ZIP} ({OUT_ZIP.stat().st_size / 1e6:.0f} MB): gene_esm {esm.shape}, run_text {txt.shape}; "
        f"/kaggle/working holds {sorted(p.name for p in WORK.iterdir())}")


main()
