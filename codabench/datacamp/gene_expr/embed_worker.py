import argparse
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

MAX_AA = 1022
TOKENS = 12_000
MAX_BATCH = 512
LOG_EVERY = 200


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", required=True)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--nshard", type=int, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time()
    df = pd.read_parquet(args.seqs).iloc[args.shard::args.nshard]
    ids = df["id"].to_numpy()
    seqs = [s[:MAX_AA] for s in df["seq"]]
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, torch_dtype=torch.float16).cuda().eval()
    special = torch.tensor([tok.cls_token_id, tok.eos_token_id, tok.pad_token_id], device="cuda")
    out = np.zeros((len(seqs), model.config.hidden_size), dtype=np.float16)
    i, n_batch, n_tok = 0, 0, 0
    with torch.inference_mode():
        while i < len(seqs):
            longest = len(seqs[i]) + 2
            j = min(len(seqs), i + max(1, min(MAX_BATCH, TOKENS // longest)))
            enc = tok(seqs[i:j], return_tensors="pt", padding=True, truncation=True, max_length=MAX_AA + 2)
            ids_t = enc["input_ids"].cuda()
            h = model(input_ids=ids_t, attention_mask=enc["attention_mask"].cuda()).last_hidden_state.float()
            m = (~torch.isin(ids_t, special)).float()
            out[i:j] = ((h * m[:, :, None]).sum(1) / m.sum(1, keepdim=True).clamp(min=1.0)).cpu().numpy()
            n_tok += int(m.sum())
            n_batch += 1
            i = j
            if n_batch % LOG_EVERY == 0 or i == len(seqs):
                el = time.time() - t0
                print(f"[esm {args.shard}] {i:,}/{len(seqs):,} seqs, {n_tok / el:,.0f} residues/s, "
                      f"elapsed {el / 60:.1f} min", flush=True)
    assert np.isfinite(out.astype(np.float32)).all(), "non-finite embedding"
    np.savez(args.out, ids=ids, emb=out)
    print(f"[esm {args.shard}] wrote {args.out} in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
