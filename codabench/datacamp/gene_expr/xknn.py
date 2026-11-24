import argparse
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common.runtime import as_text, find
from common.runtime import log as _log

log = partial(_log, tag="xknn ")

XKNN_K, CHUNK, SMOKE_CANDIDATES = 20, 512, 30_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")

    keys = pd.read_parquet(find("gene_model_keys.parquet"))
    seq = pd.read_parquet(find("gene_seq_keys.parquet"))
    lab = as_text(keys["genome_id"]) + "|" + as_text(keys["feature_id"])
    pos = pd.Index(as_text(seq["genome_id"]) + "|" + as_text(seq["feature_id"])).get_indexer(lab)
    esm_row = np.where(pos >= 0, seq["esm_row"].to_numpy(dtype=np.int64)[np.maximum(pos, 0)], -1)
    genome = pd.factorize(as_text(keys["genome_id"]))[0].astype(np.int64)
    unit = keys["unit"].to_numpy(dtype=np.int64)
    G = len(keys)

    has = np.flatnonzero(esm_row >= 0)
    has = has[np.argsort(genome[has], kind="stable")]
    cand = has
    if args.smoke:
        cand = np.sort(np.random.default_rng(0).choice(has, size=min(SMOKE_CANDIDATES, len(has)), replace=False))
        cand = cand[np.argsort(genome[cand], kind="stable")]
    E = np.load(find("esm_full.npy"), mmap_mode="r")
    sample = np.asarray(E[np.sort(np.random.default_rng(1).choice(len(E), size=min(200_000, len(E)), replace=False))],
                        dtype=np.float32)
    mu, sd = torch.as_tensor(sample.mean(0), device=dev), torch.as_tensor(sample.std(0) + 1e-6, device=dev)
    del sample

    def vectors(rows):
        Z = torch.empty(len(rows), E.shape[1], dtype=torch.float16, device=dev)
        for i in range(0, len(rows), 100_000):
            x = torch.as_tensor(np.asarray(E[esm_row[rows[i:i + 100_000]]], dtype=np.float32), device=dev)
            Z[i:i + 100_000] = torch.nn.functional.normalize((x - mu) / sd, dim=1).half()
        return Z
    Zc = vectors(cand)
    c_gen = torch.as_tensor(genome[cand], device=dev)
    c_unit = torch.as_tensor(unit[cand], device=dev)
    c_idx = torch.as_tensor(cand, device=dev)
    log(f"genes {G:,}, with a protein {len(has):,}; candidates {len(cand):,}{' (smoke)' if args.smoke else ''}")

    nbr = np.full((G, XKNN_K), -1, dtype=np.int32)
    sim = np.zeros((G, XKNN_K), dtype=np.float16)
    with torch.no_grad():
        for i in range(0, len(has), CHUNK):
            q = has[i:i + CHUNK]
            Zq = vectors(q)
            S = Zq @ Zc.T
            q_gen = torch.as_tensor(genome[q], device=dev)
            q_unit = torch.as_tensor(unit[q], device=dev)
            S.masked_fill_((q_gen[:, None] == c_gen[None]) | (q_unit[:, None] == c_unit[None]), -float("inf"))
            val, at = S.float().topk(min(XKNN_K, S.shape[1]), dim=1)
            ok = torch.isfinite(val)
            nbr[q, :val.shape[1]] = torch.where(ok, c_idx[at], -1).cpu().numpy()
            sim[q, :val.shape[1]] = torch.where(ok, val, 0.0).cpu().numpy().astype(np.float16)
            if i % (200 * CHUNK) == 0:
                log(f"{i + len(q):,} / {len(has):,} queries")

    got = nbr >= 0
    rows, cols = np.nonzero(got)
    assert not (genome[nbr[rows, cols]] == genome[rows]).any(), "a neighbour shares the gene's genome"
    assert not (unit[nbr[rows, cols]] == unit[rows]).any(), "a neighbour shares the gene's unit"
    s = sim.astype(np.float32)
    assert np.isfinite(s).all() and (np.diff(np.where(got, s, -9.0), axis=1) <= 1e-3).all(), "similarities unsorted"
    first = got[:, 0]
    log(f"genes with a neighbour {first.mean():.4f} (with a protein {first[esm_row >= 0].mean():.4f}); similarity "
        f"top-1 mean {s[first, 0].mean():.3f}, top-{XKNN_K} mean {s[got[:, -1], -1].mean():.3f}; neighbours per gene "
        f"{got.sum(1)[first].mean():.1f}")
    np.save(out / "xknn_nbr.npy", nbr)
    np.save(out / "xknn_sim.npy", sim)
    log(f"wrote {out / 'xknn_nbr.npy'}, {out / 'xknn_sim.npy'}")


if __name__ == "__main__":
    main()
