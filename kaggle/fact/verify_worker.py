import argparse
import contextlib
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from huggingface_hub import snapshot_download
from peft import LoraConfig, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from dist_common import budget_batches, dtype_kw, lr_scale, mem_gb, save_npz, say, setup, sync_max

TRAIN_TOKENS = 1792
TRAIN_CAP = 32
CHUNK = 4096
ACC = 4
INFER_TOKENS = 8192
INFER_CAP = 64
LR = 1e-4
MAXLEN = 448
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load(name, dt, dev, rank):
    if rank == 0:
        snapshot_download(name, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken"])
    dist.barrier()
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        name, num_labels=3, attn_implementation="sdpa", device_map={"": dev.index}, **dtype_kw(dt))
    model.config.pad_token_id = tok.pad_token_id
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, LoraConfig(task_type="SEQ_CLS", r=16, lora_alpha=32, lora_dropout=0.05,
                                             target_modules=LORA_TARGETS))
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    return tok, model


def collate(ids, pad, dev):
    n = max(map(len, ids))
    x = torch.full((len(ids), n), pad, dtype=torch.long)
    m = torch.zeros((len(ids), n), dtype=torch.long)
    for i, s in enumerate(ids):
        x[i, :len(s)] = torch.tensor(s)
        m[i, :len(s)] = 1
    return x.to(dev), m.to(dev)


def global_batches(idx, lens, seed, world):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(idx)
    out = []
    for a in range(0, len(idx), CHUNK):
        chunk = idx[a:a + CHUNK]
        out += budget_batches(chunk[np.argsort(-lens[chunk], kind="stable")], lens, TRAIN_TOKENS, TRAIN_CAP)
    groups = [out[i:i + world] for i in range(0, len(out) - world + 1, world)]
    return [groups[i] for i in rng.permutation(len(groups))]


def infer(model, iid, ilen, pad, rank, world, dev, use_amp):
    order = np.argsort(-ilen, kind="stable")[rank::world]
    model.eval()
    probs = []
    with torch.no_grad():
        for b in budget_batches(order, ilen, INFER_TOKENS, INFER_CAP):
            x, m = collate([iid[i] for i in b], pad, dev)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                probs.append(torch.softmax(model(input_ids=x, attention_mask=m).logits.float(), -1).cpu().numpy())
    p = np.concatenate(probs) if probs else np.zeros((0, 3), np.float32)
    assert np.isfinite(p).all(), "non-finite verifier probabilities"
    return order, p.astype(np.float32)


def pick_model(names, texts, dev, rank):
    for name, dt in zip(names, (torch.float16, torch.float32)):
        tok, model = load(name, dt, dev, rank)
        use_amp = dt == torch.float16
        ids = tok(texts, truncation=True, max_length=MAXLEN)["input_ids"]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            x, m = collate(ids[rank * 8:rank * 8 + 8], tok.pad_token_id, dev)
            finite = bool(torch.isfinite(model(input_ids=x, attention_mask=m).logits).all())
        bad = sync_max(0.0 if finite else 1.0, dev)
        say(f"verify: {name} {dt} step-0 logits finite on all ranks: {bad == 0}")
        if bad == 0:
            return name, tok, model, use_amp, ids
        del model
        gc.collect()
        torch.cuda.empty_cache()
    raise RuntimeError("every verifier model produced non-finite logits")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work")
    ap.add_argument("--models")
    ap.add_argument("--ext")
    ap.add_argument("--ext_rows", type=int, default=0)
    ap.add_argument("--deadline", type=float)
    ap.add_argument("--max_steps", type=int, default=0)
    args = ap.parse_args()
    rank, world, dev = setup()
    work = Path(args.work)
    tr = pd.read_parquet(work / "vtrain.parquet")
    ext = pd.read_parquet(args.ext)
    if args.ext_rows:
        ext = ext.iloc[:args.ext_rows]
    inf = pd.read_parquet(work / "vinfer.parquet")
    y_all = torch.tensor(np.concatenate([tr["y"].to_numpy(), ext["y"].to_numpy()]), dtype=torch.long)

    name, tok, model, use_amp, ids = pick_model(args.models.split(","), tr["text"].tolist() + ext["text"].tolist(),
                                                dev, rank)
    pad = tok.pad_token_id
    lens = np.array([len(s) for s in ids])
    cf, ex = np.arange(len(tr)), np.arange(len(tr), len(ids))
    g_ext, g1, g2 = (global_batches(ex, lens, 11, world), global_batches(cf, lens, 12, world),
                     global_batches(cf, lens, 13, world))
    mine = [g[rank] for g in g_ext + g1 + g2]
    n_opt = len(mine) // ACC
    snap = (len(g_ext) + len(g1)) // ACC
    if args.max_steps:
        n_opt = min(n_opt, args.max_steps)
        snap = n_opt // 2
    plan_tokens = sum(int(lens[b].max()) * len(b) for b in mine[:n_opt * ACC])
    iid = tok(inf["text"].tolist(), truncation=True, max_length=MAXLEN)["input_ids"]
    ilen = np.array([len(s) for s in iid])
    ddp = DDP(model, device_ids=[dev.index])
    params = [p for p in model.parameters() if p.requires_grad]
    say(f"verify: {name}, train rows {len(tr)} CFEVER x2 epochs + {len(ext)} external first, {n_opt} optimizer steps, "
        f"epoch-1 snapshot at step {snap}, trainable {sum(p.numel() for p in params) / 1e6:.1f}M")
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", init_scale=2 ** 12, enabled=use_amp)
    ce = torch.nn.CrossEntropyLoss()
    t0 = time.time()
    train_end = args.deadline
    losses, nonfinite, step, tokens = [], 0, 0, 0
    ddp.train()
    for step in range(n_opt):
        if step == snap and step > 0:
            t2 = time.time()
            order, p = infer(model, iid, ilen, pad, rank, world, dev, use_amp)
            save_npz(work / f"vp1_rank{rank}.npz", qid=inf["qid"].to_numpy()[order], probs=p)
            ddp.train()
            say(f"verify epoch-1 snapshot: inference {(time.time() - t2) / 60:.1f} min, peak mem {mem_gb():.1f}GB")
        if step == min(5, n_opt - 1) and step > 0:
            per_tok = sync_max((time.time() - t0) / tokens, dev)
            infer_est = 1.5 * ilen.sum() / world * per_tok / 3 + 300
            train_end = args.deadline - infer_est
            say(f"verify: {per_tok * 1e3:.3f} ms/token/gpu, plan needs {plan_tokens * per_tok / 60:.0f} min, "
                f"inference estimate {infer_est / 60:.0f} min, train must end in {(train_end - time.time()) / 60:.0f} min")
        budget = max(train_end - t0, 1.0)
        progress = sync_max(max(step / n_opt, (time.time() - t0) / budget), dev)
        if progress >= 1:
            break
        for g in opt.param_groups:
            g["lr"] = LR * lr_scale(progress, warm=0.03)
        for a in range(ACC):
            b = mine[step * ACC + a]
            x, m = collate([ids[i] for i in b], pad, dev)
            tokens += x.numel()
            ctx = ddp.no_sync() if a < ACC - 1 else contextlib.nullcontext()
            with ctx:
                with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    logits = ddp(input_ids=x, attention_mask=m).logits.float()
                loss = ce(logits, y_all[b].to(dev)) / ACC
                scaler.scale(loss).backward()
            lv = loss.item() * ACC
            nonfinite = nonfinite + 1 if not np.isfinite(lv) else 0
            losses.append(lv)
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        if sync_max(float(nonfinite >= 20 or (use_amp and scaler.get_scale() < 2 ** -6)), dev) > 0:
            raise RuntimeError(f"verifier diverged at step {step}")
        if step % 100 == 0:
            say(f"verify step {step}/{n_opt} loss {np.nanmean(losses[-400:]):.4f} scale {scaler.get_scale():.0f} "
                f"elapsed {(time.time() - t0) / 60:.1f} min mem {mem_gb():.1f}GB")
    say(f"verify train done: {step} steps, {(time.time() - t0) / 60:.1f} min, loss {np.nanmean(losses[-400:]):.4f}")

    del ddp, opt, scaler, params
    torch.cuda.empty_cache()
    t1 = time.time()
    order, p = infer(model, iid, ilen, pad, rank, world, dev, use_amp)
    say(f"verify inference {len(order)} rows/rank in {(time.time() - t1) / 60:.1f} min, peak mem {mem_gb():.1f}GB")
    save_npz(work / f"vp_rank{rank}.npz", qid=inf["qid"].to_numpy()[order], probs=p)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
