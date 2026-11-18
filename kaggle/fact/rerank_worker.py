import argparse
import contextlib
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from dist_common import budget_batches, dtype_kw, equal_shard, lr_scale, mem_gb, save_npz, say, setup, sync_max

N_NEG = 15
HARD_POOL = 40
POS_PER_CLAIM = 2
ACC = 4
LR = 2e-5
MAXLEN = 224
SCORE_CHARS = 48000
SCORE_CAP = 1024
TRAIN_SCORE_TOP = 24


def passage(title, text):
    return f"{title.replace('_', ' ')}：{text[:300]}"


def build_items(claims, cands, gold, rng):
    gold_by_q = {q: list(zip(d["title"], d["sid"], d["text"])) for q, d in gold.groupby("qid")}
    cand_by_q = {q: list(zip(d["title"], d["sid"], d["text"])) for q, d in cands.groupby("qid")}
    items = []
    for qid, claim, y, split in zip(claims["qid"], claims["claim"], claims["y"], claims["split"]):
        if split != "train" or y == 2 or qid not in gold_by_q:
            continue
        gk = {(t, s) for t, s, _ in gold_by_q[qid]}
        negs = [passage(t, x) for t, s, x in cand_by_q.get(qid, []) if (t, s) not in gk][:HARD_POOL]
        pos = [g for g in gold_by_q[qid] if g[2]]
        if not negs or not pos:
            continue
        for j in rng.permutation(len(pos))[:POS_PER_CLAIM]:
            items.append((claim[:100], passage(pos[j][0], pos[j][2]), negs))
    return items


def score_rows(model, tok, qs, texts, phases, deadline, rank, world, dev):
    lens = np.array([len(a) + len(b) for a, b in zip(qs, texts)])
    done_idx, done_score = [], []
    t1 = time.time()
    for phase, idx in enumerate(phases):
        idx = idx[np.argsort(-lens[idx], kind="stable")][rank::world]
        for b in budget_batches(idx, lens, SCORE_CHARS, SCORE_CAP):
            if phase == 2 and time.time() > deadline:
                say("rerank scoring: deadline reached, train rows left unscored")
                break
            batch = tok([qs[i] for i in b], [texts[i] for i in b], truncation="only_second",
                        max_length=MAXLEN, padding=True, return_tensors="pt").to(dev)
            with torch.no_grad():
                s = model(**batch).logits.view(-1).float().cpu().numpy()
            done_idx.append(b)
            done_score.append(s)
        say(f"rerank scored phase {phase} ({len(idx)} rows on this rank) at {(time.time() - t1) / 60:.1f} min, "
            f"peak mem {mem_gb():.1f}GB")
    idx = np.concatenate(done_idx) if done_idx else np.zeros(0, np.int64)
    sc = np.concatenate(done_score).astype(np.float32) if done_score else np.zeros(0, np.float32)
    assert np.isfinite(sc).all(), "non-finite reranker scores"
    return idx, sc


def train(args, claims, cands, gold, rank, world, dev):
    items = build_items(claims, cands, gold, np.random.default_rng(0))
    mine = equal_shard(len(items), rank, world, seed=1)
    n_opt = len(mine) // ACC
    if args.max_steps:
        n_opt = min(n_opt, args.max_steps)
    say(f"rerank: {len(items)} groups, world {world}, {n_opt} optimizer steps max")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=1)
    model.base_model.embeddings.requires_grad_(False)
    model.to(dev)
    ddp = DDP(model, device_ids=[dev.index])
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", init_scale=2 ** 12)
    ce = torch.nn.CrossEntropyLoss()
    srng = np.random.default_rng(100 + rank)
    t0 = time.time()
    losses, step = [], 0
    ddp.train()
    for step in range(n_opt):
        progress = sync_max(max(step / n_opt, (time.time() - t0) / args.train_budget), dev)
        if progress >= 1:
            break
        for g in opt.param_groups:
            g["lr"] = LR * lr_scale(progress)
        for a in range(ACC):
            claim, pos, negs = items[mine[step * ACC + a]]
            pick = srng.choice(len(negs), N_NEG, replace=len(negs) < N_NEG)
            batch = tok([claim] * (N_NEG + 1), [pos] + [negs[j] for j in pick], truncation="only_second",
                        max_length=MAXLEN, padding=True, return_tensors="pt").to(dev)
            ctx = ddp.no_sync() if a < ACC - 1 else contextlib.nullcontext()
            with ctx:
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = ddp(**batch).logits.view(1, -1).float()
                loss = ce(logits, torch.zeros(1, dtype=torch.long, device=dev)) / ACC
                scaler.scale(loss).backward()
            losses.append(loss.item() * ACC)
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        if step % 200 == 0 or step == 20:
            el = time.time() - t0
            say(f"rerank step {step}/{n_opt} loss {np.mean(losses[-800:]):.4f} "
                f"pairs/s/gpu {(step + 1) * ACC * (N_NEG + 1) / el:.0f} scale {scaler.get_scale():.0f} mem {mem_gb():.1f}GB")
    say(f"rerank train done: {step} steps, {(time.time() - t0) / 60:.1f} min, loss {np.mean(losses[-800:]):.4f}")
    del ddp, opt, scaler, params
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model.half().eval()
    if rank == 0:
        model.save_pretrained(Path(args.work) / "reranker")
        tok.save_pretrained(Path(args.work) / "reranker")
    return tok, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work")
    ap.add_argument("--model")
    ap.add_argument("--train_budget", type=float)
    ap.add_argument("--deadline", type=float)
    ap.add_argument("--max_steps", type=int, default=0)
    ap.add_argument("--score_file")
    args = ap.parse_args()
    rank, world, dev = setup()
    work = Path(args.work)
    if args.score_file:
        rows = pd.read_parquet(work / args.score_file)
        tok = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForSequenceClassification.from_pretrained(args.model, **dtype_kw(torch.float16)).to(dev).eval()
        idx, sc = score_rows(model, tok, rows["claim"].str[:100].tolist(),
                             [passage(t, x) for t, x in zip(rows["title"], rows["text"])],
                             [np.arange(len(rows))], None, rank, world, dev)
        save_npz(work / f"hop_rank{rank}.npz", idx=idx, score=sc)
    else:
        claims = pd.read_parquet(work / "claims.parquet")
        cands = pd.read_parquet(work / "cands.parquet")
        gold = pd.read_parquet(work / "gold.parquet")
        tok, model = train(args, claims, cands, gold, rank, world, dev)
        order = {"dev": 0, "test": 1, "train": 2}
        split_of = dict(zip(claims["qid"], claims["split"].map(order)))
        claim_of = dict(zip(claims["qid"], claims["claim"].str[:100]))
        pri = cands["qid"].map(split_of).to_numpy().copy()
        pri[(pri == 2) & (cands["rank"].to_numpy() >= TRAIN_SCORE_TOP)] = -1
        idx, sc = score_rows(model, tok, [claim_of[q] for q in cands["qid"]],
                             [passage(t, x) for t, x in zip(cands["title"], cands["text"])],
                             [np.where(pri == p)[0] for p in (0, 1, 2)], args.deadline, rank, world, dev)
        save_npz(work / f"rr_rank{rank}.npz", idx=idx, score=sc)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
