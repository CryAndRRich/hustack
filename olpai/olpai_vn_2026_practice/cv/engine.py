from __future__ import annotations

import time

import numpy as np
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler

from common.training import (EMA, amp_settings, autocast, cosine_with_warmup, make_grad_scaler,
                                 make_train_loader, shutdown_workers)
from cv.augment import ClipAugment
from cv.config import NUM_CLASSES
from cv.data import load_batch
from cv.metrics import summarize


@torch.inference_mode()
def _predict_local(model, transform, array, indices, batch_size, device, use_amp, amp_dtype, views,
                   noise_seed=None) -> tuple[np.ndarray, np.ndarray, int]:
    model.eval()
    pin = device.type == "cuda"
    out, noisy, fallbacks = [], [], 0
    for start in range(0, len(indices), batch_size):
        batch = indices[start:start + batch_size]
        n = len(batch)
        if n < batch_size:
            batch = np.concatenate([batch, np.repeat(batch[-1:], batch_size - n)])
        clips = load_batch(array, batch, pin)
        t = clips.size(1)
        frames, flags = transform.eval_frames(clips, None if noise_seed is None else noise_seed * 100_003 + int(batch[0]))
        total = 0
        for scale, flip in views:
            x = transform.eval_view(frames, t, scale, flip)
            with autocast(device, use_amp, amp_dtype):
                p = model(x).float().softmax(1)
            if not torch.isfinite(p).all():
                fallbacks += 1
                p = model(x.float()).float().softmax(1)
                bad = ~torch.isfinite(p).all(1)
                p[bad] = float("nan")
            total = total + (p[:, transform.swap] if flip else p)
        out.append((total / len(views))[:n])
        noisy.append(flags[:n])
    if not out:
        return np.empty((0, NUM_CLASSES), dtype=np.float32), np.zeros(0, dtype=bool), 0
    return torch.cat(out).cpu().numpy(), torch.cat(noisy).cpu().numpy(), fallbacks


def predict(model: nn.Module, transform: ClipAugment, array: np.ndarray, indices: np.ndarray,
            cfg, ctx, use_amp: bool, amp_dtype, views=((1.0, False),), noise_seed: int | None = None,
            return_info: bool = False):
    indices = np.asarray(indices)
    positions = np.arange(len(indices))[ctx.rank::ctx.world_size]
    local, noisy, fallbacks = _predict_local(model, transform, array, indices[positions], cfg.batch_size * 2,
                                             ctx.device, use_amp, amp_dtype, list(views), noise_seed)

    probs = np.empty((len(indices), NUM_CLASSES), dtype=np.float32)
    flags = np.zeros(len(indices), dtype=bool)
    total_fallbacks = 0
    for pos, part, part_noisy, part_fallbacks in ctx.gather((positions, local, noisy, fallbacks)):
        probs[pos] = part
        flags[pos] = part_noisy
        total_fallbacks += part_fallbacks
    if return_info:
        return probs, {"noisy": flags, "fp32_fallback": total_fallbacks}
    return probs


def _is_better(metrics: dict, best: dict) -> bool:
    if best["state"] is None or metrics["f1"] != best["f1"]:
        return best["state"] is None or metrics["f1"] > best["f1"]
    return metrics["loss"] < best["loss"]


class LogitAdjustedLoss(nn.Module):
    def __init__(self, labels: np.ndarray, tau: float, label_smoothing: float):
        super().__init__()
        counts = np.maximum(np.bincount(labels, minlength=NUM_CLASSES), 1)
        self.register_buffer("offset", tau * torch.tensor(np.log(counts / counts.sum()), dtype=torch.float32))
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return nn.functional.cross_entropy(logits.float() + self.offset, target, label_smoothing=self.label_smoothing)


def make_optimizer(model: nn.Module, cfg, device: torch.device):
    groups = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        backbone = name.startswith("backbone.")
        decay = param.ndim > 1 and not name.endswith("pos_embed")
        groups.setdefault((backbone, decay), []).append(param)
    param_groups = [{"params": params,
                     "lr": cfg.lr * (cfg.backbone_lr_scale if backbone else 1.0),
                     "weight_decay": cfg.weight_decay if decay else 0.0}
                    for (backbone, decay), params in groups.items()]
    if device.type == "cuda":
        try:
            return torch.optim.AdamW(param_groups, fused=True)
        except (TypeError, RuntimeError):
            pass
    return torch.optim.AdamW(param_groups)


def fit(cfg, ctx, model: nn.Module, train_ds, val_array, val_indices, val_labels, swap, log) -> dict:
    device = ctx.device
    use_amp, amp_dtype = amp_settings(cfg, device)
    transform = ClipAugment(cfg, device, swap)
    loader = make_train_loader(train_ds, cfg, ctx)
    ema = EMA(model, cfg.ema_decay) if cfg.ema_decay > 0 else None
    eval_model = ema.module if ema is not None else model

    train_model = (DistributedDataParallel(model, device_ids=[ctx.local_rank], gradient_as_bucket_view=True)
                   if ctx.enabled else model)
    optimizer = make_optimizer(model, cfg, device)
    steps_per_epoch = len(loader)
    scheduler = cosine_with_warmup(optimizer, steps_per_epoch * cfg.warmup_epochs,
                                   steps_per_epoch * cfg.epochs)
    scaler = make_grad_scaler(use_amp and amp_dtype == torch.float16)
    loss_fn = LogitAdjustedLoss(train_ds.labels[train_ds.indices], cfg.logit_adjust, cfg.label_smoothing).to(device)
    log(f"[train] {len(train_ds)} videos | {ctx.world_size} procs × batch {cfg.batch_size} videos "
        f"× {cfg.num_frames} frames | {steps_per_epoch} steps/epoch | AMP={'off' if not use_amp else str(amp_dtype).split('.')[-1]} "
        f"| workers/proc={cfg.num_workers} | image {cfg.image_size} | RRC {cfg.rrc_scale} | temporal {cfg.temporal_scale} "
        f"| hflip={cfg.hflip_p} (label swap {cfg.flip_pairs}) | rotate ±{cfg.rotate_deg}° | jitter {cfg.jitter} | gray {cfg.gray_p} "
        f"| synthetic noise p={cfg.noise_p} density {cfg.noise_density} | denoise={cfg.denoise} "
        f"| EMA={cfg.ema_decay or 'off'} | logit_adjust τ={cfg.logit_adjust}")

    best = {"state": None, "epoch": 0, "f1": -1.0, "loss": float("inf")}
    for epoch in range(1, cfg.epochs + 1):
        if isinstance(loader.sampler, DistributedSampler):
            loader.sampler.set_epoch(epoch)
        train_model.train()
        started = time.time()
        loss_sum = torch.zeros((), device=device)
        seen = torch.zeros((), device=device)
        for clips, labels, _ in loader:
            x, y = transform.train_batch(clips, labels.to(device, non_blocking=True))
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
        probs = predict(eval_model, transform, val_array, val_indices, cfg, ctx, use_amp, amp_dtype)
        metrics = summarize(val_labels, probs)
        improved = _is_better(metrics, best)
        if improved:
            best.update(f1=metrics["f1"], loss=metrics["loss"], acc=metrics["acc"], epoch=epoch,
                        state={k: v.detach().clone() for k, v in eval_model.state_dict().items()})
        log(f"[epoch {epoch}/{cfg.epochs}] train_loss={(loss_sum / seen.clamp(min=1)).item():.4f} "
            f"| val_loss={metrics['loss']:.4f} | val_f1={metrics['f1']:.4f} | val_acc={metrics['acc']:.4f} "
            f"| {train_time:.1f}s ({seen.item() / train_time:.1f} video/s) | eval {time.time() - started - train_time:.1f}s"
            + (" *" if improved else ""))

    model.load_state_dict(best.pop("state"))
    shutdown_workers(loader)
    return best
