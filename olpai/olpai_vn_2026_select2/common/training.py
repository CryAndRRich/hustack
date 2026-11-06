from __future__ import annotations

import copy
import math

import torch
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler

from common.utils import seed_worker


def amp_settings(cfg, device: torch.device):
    if not cfg.amp or device.type != "cuda":
        return False, torch.float32
    major, _ = torch.cuda.get_device_capability(device)
    return True, torch.bfloat16 if major >= 8 else torch.float16


def autocast(device: torch.device, enabled: bool, dtype: torch.dtype):
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


def make_grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.module = copy.deepcopy(model).requires_grad_(False)
        self.decay = decay
        self.steps = 0
        pairs = list(zip(self.module.state_dict().values(), model.state_dict().values()))
        self.float_pairs = [(e, m) for e, m in pairs if e.is_floating_point()]
        self.other_pairs = [(e, m) for e, m in pairs if not e.is_floating_point()]

    @torch.no_grad()
    def update(self) -> None:
        decay = min(self.decay, (1 + self.steps) / (10 + self.steps))
        self.steps += 1
        ema, cur = zip(*self.float_pairs)
        torch._foreach_lerp_(list(ema), list(cur), 1.0 - decay)
        for e, m in self.other_pairs:
            e.copy_(m)


def make_optimizer(model: nn.Module, cfg, device: torch.device):
    kwargs = dict(lr=cfg.lr, weight_decay=cfg.weight_decay)
    if device.type == "cuda":
        try:
            return torch.optim.AdamW(model.parameters(), fused=True, **kwargs)
        except (TypeError, RuntimeError):
            pass
    return torch.optim.AdamW(model.parameters(), **kwargs)


def cosine_with_warmup(optimizer, warmup_steps: int, total_steps: int):
    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def maybe_compile(module: nn.Module, cfg, ctx, log) -> nn.Module:
    if not cfg.compile or ctx.device.type != "cuda":
        return module
    major, minor = torch.cuda.get_device_capability(ctx.device)
    if major < 7:
        log(f"[compile] GPU sm{major}{minor} does not support Triton (e.g. P100) — skipping torch.compile")
        return module
    log(f"[compile] torch.compile(mode={cfg.compile_mode!r}) — first epoch slower due to compilation")
    return torch.compile(module, mode=cfg.compile_mode)


def make_train_loader(dataset, cfg, ctx) -> DataLoader:
    sampler = None
    if ctx.enabled:
        sampler = DistributedSampler(dataset, num_replicas=ctx.world_size, rank=ctx.rank,
                                     shuffle=True, seed=cfg.seed, drop_last=True)
    generator = torch.Generator()
    generator.manual_seed(cfg.seed + ctx.rank)
    workers = cfg.num_workers
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        drop_last=True,
        num_workers=workers,
        pin_memory=ctx.device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=workers > 0,
        prefetch_factor=4 if workers > 0 else None,
    )


def shutdown_workers(loader: DataLoader) -> None:
    iterator = getattr(loader, "_iterator", None)
    if iterator is not None and hasattr(iterator, "_shutdown_workers"):
        iterator._shutdown_workers()
    loader._iterator = None
