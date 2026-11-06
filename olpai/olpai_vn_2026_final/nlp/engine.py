from __future__ import annotations

import math
import time
from contextlib import nullcontext

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from common.training import autocast, cosine_with_warmup, make_grad_scaler, make_optimizer
from nlp.linearize import parse_output
from nlp.metric import evaluate
from nlp.model import IGNORE, Collator, Item, amp_mode, generate


def predict_texts(model, tokenizer, items: list[Item], cfg, ctx, num_beams: int) -> list[str]:
    use_amp, amp_dtype = amp_mode(cfg, ctx.device)
    positions = list(range(ctx.rank, len(items), ctx.world_size))
    local = generate(model, tokenizer, [items[i] for i in positions], cfg, ctx.device, num_beams, use_amp, amp_dtype)
    texts = [""] * len(items)
    for part_positions, part_texts in ctx.gather((positions, local)):
        for p, t in zip(part_positions, part_texts):
            texts[p] = t
    return texts


def parse_all(texts: list[str], examples, mapper, cfg) -> dict:
    return {(ex.domain, ex.index): parse_output(t, ex, mapper, cfg.aspect_fuzzy, cfg.drop_unknown_category)
            for t, ex in zip(texts, examples)}


@torch.no_grad()
def dev_loss(model, items: list[Item], collate: Collator, cfg, ctx) -> float:
    model.eval()
    use_amp, amp_dtype = amp_mode(cfg, ctx.device)
    total = torch.zeros((), device=ctx.device)
    tokens = torch.zeros((), device=ctx.device)
    local = items[ctx.rank::ctx.world_size]
    for start in range(0, len(local), cfg.batch_size * 2):
        batch = {k: v.to(ctx.device) for k, v in collate(local[start:start + cfg.batch_size * 2]).items()}
        with autocast(ctx.device, use_amp, amp_dtype):
            loss = model(**batch).loss.float()
        n = (batch["labels"] != IGNORE).sum()
        total += loss * n
        tokens += n
    return (ctx.sum(total) / ctx.sum(tokens).clamp(min=1)).item()


def fit(cfg, ctx, model, tokenizer, train_items: list[Item], dev_examples, dev_items: list[Item], mapper, log) -> dict:
    device = ctx.device
    use_amp, amp_dtype = amp_mode(cfg, device)
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.config.use_cache = False
    collate = Collator(tokenizer.pad_token_id, cfg.is_seq2seq)
    sampler = (DistributedSampler(train_items, num_replicas=ctx.world_size, rank=ctx.rank, shuffle=True,
                                  seed=cfg.seed) if ctx.enabled else None)
    generator = torch.Generator()
    generator.manual_seed(cfg.seed)
    loader = DataLoader(train_items, batch_size=cfg.batch_size, shuffle=sampler is None, sampler=sampler,
                        collate_fn=collate, generator=generator, num_workers=0, pin_memory=device.type == "cuda")

    wrapped = (DistributedDataParallel(model, device_ids=[ctx.local_rank] if device.type == "cuda" else None)
               if ctx.enabled else model)
    optimizer = make_optimizer(model, cfg, device)
    updates_per_epoch = math.ceil(len(loader) / cfg.grad_accum)
    total_updates = updates_per_epoch * cfg.epochs
    scheduler = cosine_with_warmup(optimizer, int(total_updates * cfg.warmup_ratio), total_updates)
    scaler = make_grad_scaler(use_amp and amp_dtype == torch.float16)
    log(f"[train] {len(train_items)} sentences | {ctx.world_size} procs × batch {cfg.batch_size} × accum "
        f"{cfg.grad_accum} | {updates_per_epoch} updates/epoch | AMP={str(amp_dtype).split('.')[-1] if use_amp else 'off'} "
        f"| lr={cfg.lr} | grad_ckpt={cfg.gradient_checkpointing}")

    best = {"epoch": 0, "f1": -1.0, "loss": float("inf"), "state": None}
    for epoch in range(1, cfg.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        wrapped.train()
        started = time.time()
        loss_sum = torch.zeros((), device=device)
        for step, batch in enumerate(loader):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            sync = (step + 1) % cfg.grad_accum == 0 or step + 1 == len(loader)
            with wrapped.no_sync() if ctx.enabled and not sync else nullcontext():
                with autocast(device, use_amp, amp_dtype):
                    loss = wrapped(**batch).loss
                scaler.scale(loss / cfg.grad_accum).backward()
            if sync:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            loss_sum += loss.detach().float()
        train_loss = (ctx.sum(loss_sum) / (len(loader) * ctx.world_size)).item()
        train_time = time.time() - started

        if cfg.train_on_dev or epoch < cfg.eval_from_epoch:
            log(f"[epoch {epoch}/{cfg.epochs}] train_loss={train_loss:.4f} | {train_time:.0f}s")
            continue

        loss = dev_loss(model, dev_items, collate, cfg, ctx)
        texts = predict_texts(model, tokenizer, dev_items, cfg, ctx, cfg.eval_beams)
        scores = evaluate(dev_examples, parse_all(texts, dev_examples, mapper, cfg))
        metrics = {"f1": scores["avg_f1"], "loss": loss}
        if cfg.monitor == "loss":
            improved = metrics["loss"] < best["loss"]
        else:
            improved = metrics["f1"] > best["f1"] or (metrics["f1"] == best["f1"] and metrics["loss"] < best["loss"])
        if improved:
            best.update(metrics, epoch=epoch,
                        state={k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()})
        per_domain = " ".join(f"{d[:4]}={scores[d]['f1']:.3f}" for d in scores if d != "avg_f1")
        log(f"[epoch {epoch}/{cfg.epochs}] train_loss={train_loss:.4f} | dev_loss={loss:.4f} | dev_F1={metrics['f1']:.4f} "
            f"({per_domain}) | train {train_time:.0f}s, eval {time.time() - started - train_time:.0f}s"
            + (" *" if improved else ""))

    if best["state"] is not None:
        model.load_state_dict(best.pop("state"))
    else:
        best.pop("state")
        best["epoch"] = cfg.epochs
    if cfg.gradient_checkpointing:
        model.config.use_cache = True
    return best
