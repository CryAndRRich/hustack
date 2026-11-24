import bisect
import io
import multiprocessing as mp
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common.runtime import T0, WORK, log, pip_install, run_workers, walk

TMP = Path("/tmp/gene_seq")
WORKER = Path("/tmp/gene_expr/embed_worker.py")
ESM_MODEL = "facebook/esm2_t33_650M_UR50D"
C3P_REPO = "https://github.com/dufaultc/contrastive-promoter-protein-pretraining.git"
C3P_COMMIT = "3f089f9"
C3P_HF = "dufaultc/contrastive-promoter-protein-pretraining"
C3P_MODELS = {"c3p_5m": "C3P_5M.pth", "c3p_100m": "C3P_100M.pth"}
C3P_BATCH = 512
PROM_MAX, PROM_MIN, OPERON_WINDOW = 512, 100, 20
OCCUPYING = ("gene", "CDS", "tRNA", "rRNA", "ncRNA", "tmRNA")
ENA_URL = "https://www.ebi.ac.uk/ena/portal/api/search"
ENA_FIELDS = ("run_accession", "instrument_platform", "instrument_model", "library_layout", "library_strategy",
              "library_selection", "library_source", "read_count", "base_count", "first_public",
              "library_construction_protocol")
ENA_BATCH = 200
KEYWORDS = {"kw_stranded": r"strand|dutp|directional",
            "kw_rrna_depleted": r"ribo-?zero|ribominus|microbexpress|rrna (depletion|removal)|deplet",
            "kw_total_rna": r"total rna"}
OUT_ZIP = WORK / "gene-expr-seq.zip"


def promoters_of(rec, cds):
    spans = []
    for f in rec.features:
        if f.type in OCCUPYING and f.location is not None and not {"pseudo", "pseudogene"} & set(f.qualifiers):
            spans += [(int(q.start), int(q.end)) for q in f.location.parts if int(q.end) > int(q.start)]
    by_start = sorted(spans)
    starts = [s for s, _ in by_start]
    pre_max_end = np.maximum.accumulate([e for _, e in by_start]) if by_start else np.array([])
    by_end = sorted(spans, key=lambda x: x[1])
    ends = [e for _, e in by_end]
    suf_min_start = np.minimum.accumulate([s for s, _ in by_end][::-1])[::-1] if by_end else np.array([])
    seq, n = rec.seq, len(rec.seq)
    out = [("", False, 0)] * len(cds)
    for strand in (1, -1):
        idx = [i for i, c in enumerate(cds) if c[2] == strand and not c[3]]
        idx.sort(key=lambda i: cds[i][0] if strand == 1 else -cds[i][1])
        prev = None
        for i in idx:
            s, e = cds[i][:2]
            if strand == 1:
                k = bisect.bisect_left(starts, s)
                near = int(pre_max_end[k - 1]) if k > 0 else 0
                gap = s - near
            else:
                k = bisect.bisect_right(ends, e)
                near = int(suf_min_start[k]) if k < len(ends) else n
                gap = near - e
            gap = min(gap, PROM_MAX)
            if gap >= PROM_MIN:
                region = seq[s - gap:s] if strand == 1 else seq[e:e + gap].reverse_complement()
                out[i] = (str(region).upper(), True, gap)
            elif (prev is not None and -OPERON_WINDOW < gap <= OPERON_WINDOW and out[prev][0]
                  and (cds[prev][1] == near if strand == 1 else cds[prev][0] == near)):
                out[i] = (out[prev][0], False, gap)
            else:
                out[i] = ("", False, gap)
            prev = i
    return out


def parse_genome(path):
    from Bio import SeqIO
    recs = list(SeqIO.parse(str(path), "genbank"))
    any_translation = any("translation" in f.qualifiers for r in recs for f in r.features if f.type == "CDS")
    rows = []
    for rec in recs:
        feats = [f for f in rec.features if f.type == "CDS" and f.location is not None]
        cds = [(int(f.location.start), int(f.location.end), -1 if f.location.strand == -1 else 1,
                int(f.location.end) - int(f.location.start) > len(f.location) + 1000) for f in feats]
        try:
            has_seq = len(rec.seq) > 0 and set(str(rec.seq[:1000]).upper()) <= set("ACGTNRYKMSWBDHV")
        except Exception:
            has_seq = False
        proms = promoters_of(rec, cds) if has_seq else [("", False, 0)] * len(cds)
        for f, c, p in zip(feats, cds, proms):
            prot, src = f.qualifiers.get("translation", [""])[0], 1
            if not prot and not any_translation and has_seq and "pseudo" not in f.qualifiers:
                table = int(f.qualifiers.get("transl_table", ["11"])[0])
                try:
                    prot = str(f.extract(rec.seq).translate(table=table, to_stop=True))
                    prot = "M" + prot[1:] if prot else ""
                    src = 2
                except Exception:
                    prot = ""
            rows.append((path.stem, rec.id, rec.name, c[0] + 1, "-" if c[2] == -1 else "+", prot.rstrip("*"),
                         src if prot else 0, p[0], p[1], p[2]))
    return pd.DataFrame(rows, columns=["genome_id", "chr_id", "chr_name", "start", "strand", "seq", "prot_src",
                                       "promoter", "prom_own", "up_gap"])


def gene_sequences(feat):
    F = feat.drop_duplicates(["genome_id", "feature_id"]).reset_index(drop=True)
    F[["genome_id", "feature_id"]] = F[["genome_id", "feature_id"]].astype(str)
    F["chr"] = F["Chr"].astype(str).str.split(";").str[0]
    F["start"] = pd.to_numeric(F["Start"].astype(str).str.split(";").str[0], errors="coerce")
    F["strand"] = F["Strand"].astype(str).str.split(";").str[0]
    files = {p.stem: p for p in walk(".gbff")}
    paths = [files[g] for g in sorted(F["genome_id"].unique()) if g in files]
    with mp.get_context("fork").Pool(os.cpu_count()) as pool:
        gb = pd.concat(pool.map(parse_genome, paths, chunksize=1), ignore_index=True)
    gb["start"] = gb["start"].astype(float)
    pos = ["genome_id", "chr", "start", "strand"]
    cols = ["seq", "prot_src", "promoter", "prom_own", "up_gap"]
    got = F[pos].merge(gb.rename(columns={"chr_id": "chr"}).drop(columns="chr_name").drop_duplicates(pos),
                       on=pos, how="left")[cols]
    miss = got["seq"].isna().to_numpy()
    alt = F.loc[miss, pos].merge(gb.rename(columns={"chr_name": "chr"}).drop(columns="chr_id").drop_duplicates(pos),
                                 on=pos, how="left")[cols]
    got.loc[miss, cols] = alt.to_numpy()
    matched = got["seq"].notna().to_numpy()
    out = F[["genome_id", "feature_id"]].copy()
    out["seq"] = got["seq"].fillna("").astype(str).to_numpy()
    out["prot_src"] = got["prot_src"].fillna(0).astype(np.int8).to_numpy()
    out["promoter"] = got["promoter"].fillna("").astype(str).to_numpy()
    out["prom_own"] = got["prom_own"].fillna(False).astype(bool).to_numpy()
    out["up_gap"] = got["up_gap"].fillna(0).astype(np.int32).to_numpy()
    out["matched"] = matched
    has_p, has_q = out["seq"] != "", out["promoter"] != ""
    log(f"genes {len(out):,}: matched {matched.mean():.4f}; protein {has_p.mean():.4f} (translated "
        f"{np.mean(out['prot_src'] == 2):.4f}); promoter {has_q.mean():.4f} (own {out['prom_own'].mean():.4f}); "
        f"promoter length {np.quantile(out.loc[has_q, 'promoter'].str.len(), [0, .5, 1]).round().tolist()}")
    per_genome = out.assign(p=has_p, q=has_q).groupby("genome_id")[["p", "q"]].mean()
    log(f"genomes with < 50 % proteins: {int((per_genome['p'] < .5).sum())}, with < 20 % promoters: "
        f"{int((per_genome['q'] < .2).sum())}")
    return out


def esm_unique(seqs, tmp):
    from huggingface_hub import snapshot_download
    snapshot_download(ESM_MODEL)
    seqs = pd.Series(seqs)
    codes, uniq = pd.factorize(seqs.where(seqs != ""))
    order = np.argsort(-np.array([len(s) for s in uniq]), kind="stable")
    tmp.mkdir(parents=True, exist_ok=True)
    path = tmp / "seqs.parquet"
    pd.DataFrame({"id": order, "seq": np.asarray(uniq, dtype=object)[order]}).to_parquet(path)
    log(f"unique proteins {len(uniq):,} ({sum(len(s) for s in uniq):,} residues) for {np.sum(codes >= 0):,} genes")
    gpus = list(range(torch.cuda.device_count()))
    assert gpus, "no GPU: switch the accelerator to GPU T4 x 2"
    run_workers([([sys.executable, WORKER, "--seqs", path, "--shard", i, "--nshard", len(gpus), "--model", ESM_MODEL,
                   "--out", tmp / f"esm_{i}.npz"], i, f"[gpu {i}]") for i in gpus])
    E = None
    for i in gpus:
        z = np.load(tmp / f"esm_{i}.npz", allow_pickle=True)
        if E is None:
            E = np.zeros((len(uniq), z["emb"].shape[1]), dtype=np.float32)
        E[z["ids"]] = z["emb"].astype(np.float32)
    return E, codes.astype(np.int32)


def esm_embeddings(seqs):
    E, codes = esm_unique(seqs, TMP)
    E = E.astype(np.float16)
    assert np.isfinite(E.astype(np.float32)).all()
    return E, codes


def c3p_embeddings(promoters):
    from huggingface_hub import hf_hub_download
    src = TMP / "c3p"
    if not src.exists():
        subprocess.run(["git", "clone", "-q", C3P_REPO, str(src)], check=True)
        subprocess.run(["git", "-C", str(src), "checkout", "-q", C3P_COMMIT], check=True)
    sys.path.insert(0, str(src))
    from promoter_protein_clip.model_transformer import SequenceCLIPTransformer
    dev = torch.device("cuda:0")
    out = {}
    for name, fname in C3P_MODELS.items():
        ck = torch.load(hf_hub_download(C3P_HF, fname), map_location=dev, weights_only=False)
        a = ck["args"]
        model = SequenceCLIPTransformer(protein_model_name=a.protein_model, promoter_embedding_dim=a.promoter_embedding_dim,
                                        depth=a.transformer_depth, num_heads=a.transformer_num_heads,
                                        max_promoter_length=a.max_promoter_length, projection_dim=a.projection_dim,
                                        device=dev)
        state = {k.removeprefix("_orig_mod."): v for k, v in ck["model_state_dict"].items()}
        model.load_state_dict(state, strict=True)
        model.eval()
        L, rev = a.max_promoter_length, bool(getattr(a, "reverse_promoter", False))
        crop = [p[::-1][:L] if rev else p[-L:] for p in promoters]
        embs = []
        with torch.no_grad():
            for i in range(0, len(crop), C3P_BATCH):
                tok = model.promoter_tokenizer(crop[i:i + C3P_BATCH], padding=True, truncation=True,
                                               return_tensors="pt").to(dev)
                embs.append(model.promoter_encoder(tok).float().cpu().numpy().astype(np.float16))
        E = np.vstack(embs)
        assert np.isfinite(E.astype(np.float32)).all(), f"{name}: non-finite embedding"
        log(f"{name}: {len(E):,} promoters x {E.shape[1]} dims (max length {L}, reversed {rev}, protein side "
            f"{a.protein_model}); {time.time() - T0:.0f}s")
        out[name] = E
        del model, ck
        torch.cuda.empty_cache()
    return out


def ena_query(accs):
    data = urllib.parse.urlencode({"result": "read_run", "format": "tsv", "fields": ",".join(ENA_FIELDS),
                                   "query": " OR ".join(f"run_accession={a}" for a in accs)}).encode()
    for attempt in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(ENA_URL, data=data), timeout=120) as r:
                return pd.read_csv(io.StringIO(r.read().decode()), sep="\t", dtype=str)
        except Exception as e:
            log(f"ENA batch failed ({e!r}), retry {attempt + 1}")
            time.sleep(5 * (attempt + 1))
    raise RuntimeError("ENA unreachable: switch Internet on")


def sra_metadata(cond):
    runs = cond[["genome_id", "run_accession"]].astype(str).drop_duplicates().reset_index(drop=True)
    accs = sorted(a for a in runs["run_accession"].unique() if re.fullmatch(r"[SED]RR\d+", a))
    got = pd.concat([ena_query(accs[i:i + ENA_BATCH]) for i in range(0, len(accs), ENA_BATCH)], ignore_index=True)
    got = got.drop_duplicates("run_accession")
    reads, bases = (pd.to_numeric(got[c], errors="coerce") for c in ("read_count", "base_count"))
    got["read_length"] = (bases / reads).where(reads > 0)
    got["year"] = pd.to_numeric(got["first_public"].str[:4], errors="coerce")
    proto = got["library_construction_protocol"].fillna("").str.lower()
    for k, rx in KEYWORDS.items():
        got[k] = proto.str.contains(rx, regex=True)
    got["has_protocol"] = proto.str.len() > 0
    got = got.drop(columns=["read_count", "base_count", "first_public", "library_construction_protocol"])
    out = runs.merge(got, on="run_accession", how="left")
    found = out["instrument_platform"].notna()
    log(f"runs {len(out):,}: SRA-style accessions {len(accs):,}, found in ENA {found.sum():,}; "
        f"layout {out['library_layout'].value_counts(dropna=False).to_dict()}; "
        f"selection top {out['library_selection'].value_counts().head(5).to_dict()}; "
        f"read length median {out['read_length'].median():.0f}; "
        + ", ".join(f"{k} {out[k].fillna(False).astype(bool).mean():.3f}" for k in list(KEYWORDS) + ["has_protocol"]))
    return out


def main():
    pip_install("Bio", "biopython")
    feat = pd.read_csv(next(walk("feature_annotation.csv")), low_memory=False)
    cond = pd.read_csv(next(walk("condition_annotation.csv")), low_memory=False)
    sra = sra_metadata(cond)
    genes = gene_sequences(feat)
    esm, esm_row = esm_embeddings(genes["seq"].to_numpy())
    prom_codes, prom_uniq = pd.factorize(genes["promoter"].where(genes["promoter"] != ""))
    c3p = c3p_embeddings(list(prom_uniq))
    keys = genes[["genome_id", "feature_id", "prot_src", "prom_own", "up_gap", "matched"]].copy()
    keys["esm_row"] = esm_row
    keys["prom_row"] = prom_codes.astype(np.int32)
    keys["prom_len"] = genes["promoter"].str.len().astype(np.int16)
    TMP.mkdir(parents=True, exist_ok=True)
    np.save(TMP / "esm_full.npy", esm)
    for name, E in c3p.items():
        np.save(TMP / f"{name}.npy", E)
    pd.DataFrame({"promoter": np.asarray(prom_uniq, dtype=object)}).to_parquet(TMP / "promoters.parquet")
    keys.to_parquet(TMP / "gene_seq_keys.parquet", index=False)
    sra.to_parquet(TMP / "sra_runs.parquet", index=False)
    names = ["esm_full.npy", *(f"{n}.npy" for n in c3p), "promoters.parquet", "gene_seq_keys.parquet",
             "sra_runs.parquet"]
    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_STORED) as z:
        for name in names:
            z.write(TMP / name, name)
    shutil.rmtree(TMP, ignore_errors=True)
    shutil.rmtree(WORKER.parent, ignore_errors=True)
    log(f"wrote {OUT_ZIP} ({OUT_ZIP.stat().st_size / 1e6:.0f} MB): esm {esm.shape}, "
        + ", ".join(f"{n} {E.shape}" for n, E in c3p.items())
        + f"; /kaggle/working holds {sorted(p.name for p in WORK.iterdir())}")


if __name__ == "__main__":
    main()
