from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler

from common.training import (EMA, amp_settings, autocast, cosine_with_warmup, make_grad_scaler,
                                 make_optimizer, make_train_loader, maybe_compile, shutdown_workers)
from cv.augment import GpuAugment, make_transforms
from cv.data import load_batch
from cv.metrics import NUM_CLASSES, summarize


@torch.inference_mode()
def _predict_local(model, transform, array, indices, batch_size, device, use_amp, amp_dtype, tta) -> np.ndarray:
    model.eval()
    pin = device.type == "cuda"
    out = []
    for start in range(0, len(indices), batch_size):
        batch = indices[start:start + batch_size]
        n = len(batch)
        if n < batch_size:
            batch = np.concatenate([batch, np.repeat(batch[-1:], batch_size - n)])
        x = transform(load_batch(array, batch, pin))
        with autocast(device, use_amp, amp_dtype):
            p = model(x).float().softmax(1)
            if tta:
                p = (p + model(x.flip(3)).float().softmax(1)) / 2
        out.append(p[:n])
    return torch.cat(out).cpu().numpy() if out else np.empty((0, NUM_CLASSES), dtype=np.float32)


def predict(model: nn.Module, transform: GpuAugment, array: np.ndarray, indices: np.ndarray,
            cfg, ctx, use_amp: bool, amp_dtype, tta: bool) -> np.ndarray:
    indices = np.asarray(indices)
    positions = np.arange(len(indices))[ctx.rank::ctx.world_size]
    local = _predict_local(model, transform, array, indices[positions], cfg.batch_size * 2,
                           ctx.device, use_amp, amp_dtype, tta)

    probs = np.empty((len(indices), NUM_CLASSES), dtype=np.float32)
    for pos, part in ctx.gather((positions, local)):
        probs[pos] = part
    return probs


def _is_better(metrics: dict, best: dict, monitor: str) -> bool:
    if best["state"] is None:
        return True
    if monitor == "loss":
        return metrics["loss"] < best["loss"]
    if metrics["f1"] != best["f1"]:
        return metrics["f1"] > best["f1"]
    return metrics["loss"] < best["loss"]


def make_loss(cfg, labels: np.ndarray, device: torch.device) -> nn.Module:
    weight = None
    if cfg.class_weighted:
        counts = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float64)
        w = 1 / np.sqrt(np.maximum(counts, 1))
        weight = torch.tensor(w / w.mean(), dtype=torch.float32, device=device)
    return nn.CrossEntropyLoss(weight=weight, label_smoothing=cfg.label_smoothing)


def fit(cfg, ctx, model: nn.Module, train_ds, val_array, val_indices, val_labels, log) -> dict:
    device = ctx.device
    use_amp, amp_dtype = amp_settings(cfg, device)
    train_tf, eval_tf = make_transforms(cfg, device)
    loader = make_train_loader(train_ds, cfg, ctx)
    ema = EMA(model, cfg.ema_decay) if cfg.ema_decay > 0 else None
    eval_model = ema.module if ema is not None else model

    wrapped = (DistributedDataParallel(model, device_ids=[ctx.local_rank] if device.type == "cuda" else None,
                                       gradient_as_bucket_view=True)
               if ctx.enabled else model)
    train_model = maybe_compile(wrapped, cfg, ctx, log)
    optimizer = make_optimizer(model, cfg, device)
    steps_per_epoch = len(loader)
    scheduler = cosine_with_warmup(optimizer, steps_per_epoch * cfg.warmup_epochs,
                                   steps_per_epoch * cfg.epochs)
    scaler = make_grad_scaler(use_amp and amp_dtype == torch.float16)
    loss_fn = make_loss(cfg, train_ds.labels[train_ds.indices], device)
    h, w = cfg.input_hw
    log(f"[train] {len(train_ds)} images | {ctx.world_size} procs × batch {cfg.batch_size} "
        f"| {steps_per_epoch} steps/epoch | AMP={'off' if not use_amp else str(amp_dtype).split('.')[-1]} "
        f"| workers/proc={cfg.num_workers} | image {h}x{w} | RRC {cfg.rrc_scale} "
        f"| EMA={cfg.ema_decay or 'off'} | class_weighted={cfg.class_weighted}")

    best = {"state": None, "epoch": 0, "f1": -1.0, "loss": float("inf")}
    for epoch in range(1, cfg.epochs + 1):
        if isinstance(loader.sampler, DistributedSampler):
            loader.sampler.set_epoch(epoch)
        train_model.train()
        started = time.time()
        loss_sum = torch.zeros((), device=device)
        seen = torch.zeros((), device=device)
        for images, labels, _ in loader:
            x = train_tf(images)
            y = labels.to(device, non_blocking=True)
            with autocast(device, use_amp, amp_dtype):
                loss = loss_fn(train_model(x), y)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            if ema is not None:
                ema.update()
            loss_sum += loss.detach() * y.size(0)
            seen += y.size(0)
        loss_sum, seen = ctx.sum(loss_sum), ctx.sum(seen)
        train_time = time.time() - started

        ctx.broadcast_buffers(eval_model)
        probs = predict(eval_model, eval_tf, val_array, val_indices, cfg, ctx, use_amp, amp_dtype, tta=False)
        metrics = summarize(val_labels, probs)
        improved = _is_better(metrics, best, cfg.monitor)
        if improved:
            best.update(f1=metrics["f1"], loss=metrics["loss"], acc=metrics["acc"], epoch=epoch,
                        state={k: v.detach().clone() for k, v in eval_model.state_dict().items()})
        log(f"[epoch {epoch}/{cfg.epochs}] train_loss={(loss_sum / seen.clamp(min=1)).item():.4f} "
            f"| val_loss={metrics['loss']:.4f} | val_f1={metrics['f1']:.4f} | val_acc={metrics['acc']:.4f} "
            f"| {train_time:.1f}s ({seen.item() / train_time:.0f} img/s) | eval {time.time() - started - train_time:.1f}s"
            + (" *" if improved else ""))

    model.load_state_dict(best.pop("state"))
    shutdown_workers(loader)
    return best
