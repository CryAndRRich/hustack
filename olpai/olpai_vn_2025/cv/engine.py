from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from common.training import EMA, amp_settings, autocast, cosine_with_warmup, make_grad_scaler
from cv.augment import GpuAugment
from cv.config import NUM_CLASSES


def make_optimizer(model: nn.Module, cfg, device: torch.device):
    decay, no_decay = [], []
    for param in model.parameters():
        if param.requires_grad:
            (decay if param.ndim > 1 else no_decay).append(param)
    groups = [{"params": decay, "weight_decay": cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]
    if device.type == "cuda":
        try:
            return torch.optim.AdamW(groups, lr=cfg.lr, fused=True)
        except (TypeError, RuntimeError):
            pass
    return torch.optim.AdamW(groups, lr=cfg.lr)


def soft_targets(cfg, device: torch.device) -> torch.Tensor:
    k = NUM_CLASSES
    table = torch.full((k, k), cfg.label_smoothing / k)
    for c in range(k):
        near = [j for j in (c - 1, c + 1) if 0 <= j < k]
        table[c, c] += 1 - cfg.label_smoothing - cfg.neighbor_smoothing
        for j in near:
            table[c, j] += cfg.neighbor_smoothing / len(near)
    return table.to(device)


@torch.inference_mode()
def predict(model: nn.Module, store, rows: np.ndarray, aug: GpuAugment, batch_size: int, use_amp: bool,
            amp_dtype, views: list[int]) -> np.ndarray:
    model.eval()
    device = aug.device
    rows = np.asarray(rows, dtype=np.int64)
    out = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        n = len(batch)
        if n < batch_size:
            batch = np.concatenate([batch, np.repeat(batch[-1:], batch_size - n, axis=0)])
        base = aug.eval_base(store.get(torch.from_numpy(batch).to(device)))
        total = 0
        for k in views:
            x = aug.view(base, k)
            with autocast(device, use_amp, amp_dtype):
                p = model(x).float().softmax(1)
            if not torch.isfinite(p).all():
                p = model(x).float().softmax(1)
                p[~torch.isfinite(p).all(1)] = float("nan")
            total = total + p
        out.append((total / len(views))[:n])
    if not out:
        return np.empty((0, NUM_CLASSES), dtype=np.float32)
    return torch.cat(out).cpu().numpy()


def fit(cfg, model: nn.Module, store, train_rows: np.ndarray, train_labels: np.ndarray, val_rows: np.ndarray,
        val_labels: np.ndarray, predict_images: int, deadline: float, seed: int, log) -> tuple[nn.Module, dict]:
    device = next(model.parameters()).device
    use_amp, amp_dtype = amp_settings(cfg, device)
    aug = GpuAugment(cfg, device)
    n, bs = len(train_rows), cfg.batch_size
    steps = n // bs
    optimizer = make_optimizer(model, cfg, device)
    scheduler = cosine_with_warmup(optimizer, int(steps * cfg.warmup_epochs), steps * cfg.epochs)
    scaler = make_grad_scaler(use_amp and amp_dtype == torch.float16)
    ema = EMA(model, cfg.ema_decay) if cfg.ema_decay > 0 else None
    eval_model = ema.module if ema is not None else model
    rows_t = torch.as_tensor(train_rows, dtype=torch.long, device=device)
    targets = soft_targets(cfg, device)
    labels_t = torch.as_tensor(train_labels, dtype=torch.long, device=device)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    predict_factor = 1.3 * predict_images * len(aug.views()) / (3 * max(1, steps * bs))

    info = {"epochs_done": 0, "stopped_early": False, "epoch_time_s": None, "train_img_per_s": None, "val_curve": []}
    for epoch in range(1, cfg.epochs + 1):
        epoch_time = info["epoch_time_s"]
        if epoch_time is not None and time.time() + epoch_time * (1 + predict_factor) > deadline:
            info["stopped_early"] = True
            log(f"  EARLY STOP before epoch {epoch}/{cfg.epochs}: not enough time before deadline (epoch ~{epoch_time:.0f}s)")
            break
        model.train()
        started = time.time()
        loss_sum = torch.zeros((), device=device)
        perm = torch.randperm(n, device=device, generator=generator)
        for step in range(steps):
            index = perm[step * bs:(step + 1) * bs]
            x = aug.train(store.get(rows_t[index]))
            with autocast(device, use_amp, amp_dtype):
                logits = model(x)
            loss = -(targets[labels_t[index]] * F.log_softmax(logits.float(), 1)).sum(1).mean()
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            if ema is not None:
                ema.update()
            loss_sum += loss.detach()
        train_loss = loss_sum.item() / max(1, steps)
        info.update(epochs_done=epoch, epoch_time_s=time.time() - started)
        info["train_img_per_s"] = steps * bs / info["epoch_time_s"]
        message = (f"  epoch {epoch}/{cfg.epochs} loss={train_loss:.4f} | {info['epoch_time_s']:.0f}s "
                   f"({info['train_img_per_s']:.0f} img/s)")
        if epoch % cfg.eval_every == 0 or epoch == cfg.epochs:
            probs = predict(eval_model, store, val_rows, aug, bs * 2, use_amp, amp_dtype, views=[0])
            acc = float((probs.argmax(1) == val_labels).mean())
            info["val_curve"].append((epoch, round(acc, 4)))
            message += f" | val acc {acc:.4f} ({'EMA, ' if ema is not None else ''}no TTA)"
        log(message)
    return eval_model, info
