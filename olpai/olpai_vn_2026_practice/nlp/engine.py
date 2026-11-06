from __future__ import annotations

import random
import time

import torch
import torch.nn.functional as F

from common.training import amp_settings, autocast, make_grad_scaler
from nlp.metric import bleu_stats
from nlp.model import generate
from nlp.tokenizer import BOS, EOS, PAD, decode


def make_batches(lengths: list[int], max_tokens: int, rng: random.Random) -> list[list[int]]:
    order = sorted(range(len(lengths)), key=lambda i: lengths[i] + rng.random())
    batches, current, longest = [], [], 0
    for i in order:
        longest_next = max(longest, lengths[i])
        if current and longest_next * (len(current) + 1) > max_tokens:
            batches.append(current)
            current, longest_next = [], lengths[i]
        current.append(i)
        longest = longest_next
    if current:
        batches.append(current)
    rng.shuffle(batches)
    return batches


def pad_batch(seqs: list[list[int]], device: torch.device) -> torch.Tensor:
    width = max(map(len, seqs))
    return torch.tensor([s + [PAD] * (width - len(s)) for s in seqs], dtype=torch.long).to(device, non_blocking=True)


def inverse_sqrt(optimizer, warmup: int):
    def factor(step: int) -> float:
        step += 1
        return step / warmup if step < warmup else (warmup / step) ** 0.5

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def make_optimizer(model, cfg, device):
    decay = [p for p in model.parameters() if p.ndim > 1]
    no_decay = [p for p in model.parameters() if p.ndim <= 1]
    groups = [{"params": decay, "weight_decay": cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    kwargs = dict(lr=cfg.lr, betas=(0.9, 0.98), eps=1e-8)
    if device.type == "cuda":
        try:
            return torch.optim.AdamW(groups, fused=True, **kwargs)
        except (TypeError, RuntimeError):
            pass
    return torch.optim.AdamW(groups, **kwargs)


def translate_ids(models: list, src_ids: list[list[list[int]]], cfg, device, beam: int,
                  len_penalty: float | None = None, weights=None) -> list:
    use_amp, amp_dtype = amp_settings(cfg, device)
    for model in models:
        model.eval()
    lp = cfg.len_penalty if len_penalty is None else len_penalty
    n = len(src_ids[0])
    order = sorted(range(n), key=lambda i: max(len(ids[i]) for ids in src_ids))
    out = [None] * n
    size = max(1, cfg.gen_rows // beam)
    for start in range(0, n, size):
        chunk = order[start:start + size]
        srcs = [pad_batch([ids[i] + [EOS] for i in chunk], device) for ids in src_ids]
        with autocast(device, use_amp, amp_dtype):
            hyps = generate(list(models), srcs, beam, lp, cfg.max_len_a, cfg.max_len_b, weights)
        for i, hyp in zip(chunk, hyps):
            out[i] = hyp
    return out


def evaluate(models: list, src_ids: list[list[list[int]]], refs: list[str], sp_vi, cfg, device, beam: int,
             len_penalty: float | None = None, weights=None) -> tuple[dict, list[str]]:
    hyps = decode(sp_vi, translate_ids(models, src_ids, cfg, device, beam, len_penalty, weights))
    return bleu_stats(hyps, refs), hyps


def average_states(states: list[dict]) -> dict:
    out = {}
    for key in states[0]:
        values = [s[key] for s in states]
        out[key] = (torch.stack([v.float() for v in values]).mean(0).to(values[0].dtype)
                    if values[0].is_floating_point() else values[0].clone())
    return out


def fit(cfg, model, train_src, train_tgt, val_src, val_refs, sp_vi, device, log, resample=None) -> dict:
    use_amp, amp_dtype = amp_settings(cfg, device)
    optimizer = make_optimizer(model, cfg, device)
    scheduler = inverse_sqrt(optimizer, cfg.warmup_steps)
    scaler = make_grad_scaler(use_amp and amp_dtype == torch.float16)
    rng = random.Random(cfg.seed + cfg.run_seed)
    lengths = [max(len(s) + 1, len(t) + 1) for s, t in zip(train_src, train_tgt)]
    n_batches = len(make_batches(lengths, cfg.batch_tokens, random.Random(0)))
    log(f"[train] {len(train_src)} pairs | ~{n_batches} batch/epoch ({cfg.batch_tokens} token) | "
        f"label smoothing {cfg.label_smoothing} | "
        f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params | "
        f"AMP={'off' if not use_amp else str(amp_dtype).split('.')[-1]} | lr {cfg.lr} warmup {cfg.warmup_steps}")

    top, bad_evals, best_bleu = [], 0, -1.0
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        started = time.time()
        loss_sum = torch.zeros((), device=device)
        tokens = 0
        if resample is not None:
            train_src, train_tgt = resample(epoch)
            lengths = [max(len(s) + 1, len(t) + 1) for s, t in zip(train_src, train_tgt)]
        for batch in make_batches(lengths, cfg.batch_tokens, rng):
            src = pad_batch([train_src[i] + [EOS] for i in batch], device)
            tgt_in = pad_batch([[BOS] + train_tgt[i] for i in batch], device)
            tgt_out = pad_batch([train_tgt[i] + [EOS] for i in batch], device)
            n_tok = sum(len(train_tgt[i]) + 1 for i in batch)
            with autocast(device, use_amp, amp_dtype):
                logits = model(src, tgt_in)
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), tgt_out.view(-1), ignore_index=PAD,
                                   label_smoothing=cfg.label_smoothing, reduction="sum") / n_tok
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if cfg.clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            loss_sum += loss.detach() * n_tok
            tokens += n_tok
        elapsed = time.time() - started
        message = (f"[epoch {epoch}/{cfg.epochs}] loss={loss_sum.item() / tokens:.4f} "
                   f"| lr={scheduler.get_last_lr()[0]:.2e} | {elapsed:.1f}s ({tokens / elapsed:.0f} token/s)")

        if epoch >= cfg.eval_from and (epoch % cfg.eval_every == 0 or epoch == cfg.epochs):
            bleu = evaluate([model], [val_src], val_refs, sp_vi, cfg, device, beam=1)[0]["bleu"]
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            top = sorted(top + [(bleu, epoch, state)], key=lambda x: -x[0])[:cfg.average_top]
            improved = bleu > best_bleu
            best_bleu = max(best_bleu, bleu)
            bad_evals = 0 if improved else bad_evals + 1
            message += f" | val BLEU (greedy)={bleu:.2f}" + (" *" if improved else "")
            log(message)
            if bad_evals >= cfg.patience:
                log(f"[train] early stop: {cfg.patience} evals without improvement")
                break
        else:
            log(message)
    log(f"[train] top-{len(top)} by val BLEU (greedy): {[(round(b, 2), e) for b, e, _ in top]}")
    return {"best": top[0], "top": top}
