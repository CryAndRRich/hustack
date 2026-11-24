import math
import random
import time
import traceback

import numpy as np
import torch

from common.models import MEAN, STD, param_groups


def seed_worker(worker_id):
    s = torch.initial_seed() % 2**32
    random.seed(s)
    np.random.seed(s)


def targets(ids, dev, semantic=False):
    masks, labels, keep = [], [], []
    for b in range(ids.shape[0]):
        idb = ids[b].to(dev, non_blocking=True)
        u = torch.unique(idb)
        u = u[u > 0]
        if u.numel():
            masks.append((idb[None] == u[:, None, None]).float())
            labels.append((u - 1).long() if semantic else torch.zeros(u.numel(), dtype=torch.long, device=dev))
            keep.append(b)
    return masks, labels, keep


def crowded_targets(n, h, w, dev):
    g = math.ceil(math.sqrt(n))
    k = torch.arange(n, device=dev)
    ys, xs = (k // g * (h // g))[:, None, None], (k % g * (w // g))[:, None, None]
    yy, xx = torch.arange(h, device=dev)[None, :, None], torch.arange(w, device=dev)[None, None, :]
    return ((yy >= ys) & (yy < ys + h // g) & (xx >= xs) & (xx < xs + w // g)).float()


class Averager:
    def __init__(self):
        self.n, self.avg = 0, None

    @torch.no_grad()
    def add(self, model):
        cur = {k: p.detach().float().cpu() for k, p in model.named_parameters()}
        if self.avg is None:
            self.avg = cur
        else:
            for k, v in cur.items():
                self.avg[k].mul_(self.n / (self.n + 1)).add_(v, alpha=1 / (self.n + 1))
        self.n += 1

    @torch.no_grad()
    def load_into(self, model):
        for k, p in model.named_parameters():
            p.copy_(self.avg[k].to(p.device))


def lr_factor(frac, step, cfg):
    warm = min(1.0, (step + 1) / cfg.warmup_steps)
    u = frac * cfg.swa_cycles - math.floor(frac * cfg.swa_cycles)
    return warm * (cfg.swa_min_frac + (1 - cfg.swa_min_frac) * 0.5 * (1 + math.cos(math.pi * u)))


def train(cfg, model, kind, dataset, batch, accum, hours, lr, mem_limit, dev, smoke, log, on_cycle=None, semantic=False):
    dl = torch.utils.data.DataLoader(dataset, batch_size=batch, shuffle=True, num_workers=cfg.num_workers,
                                     worker_init_fn=seed_worker, drop_last=True, pin_memory=True,
                                     persistent_workers=cfg.num_workers > 0)
    groups, summary = param_groups(model, kind, lr, cfg)
    log(f"train {kind}: param groups {summary}; fp32 loss patch {getattr(model, 'loss_fp32', None)}; "
        f"weights {next(model.parameters()).dtype}; {len(dataset)} items, batch {batch} x accum {accum}, {hours:.2f} h")
    assert next(model.parameters()).dtype == torch.float32
    opt = torch.optim.AdamW(groups, lr=lr, weight_decay=cfg.weight_decay)
    params = [p for g in groups for p in g["params"]]
    scaler = torch.amp.GradScaler("cuda")
    mean, std = MEAN.to(dev), STD.to(dev)
    budget = max(hours * 3600, 1.0)
    swa = Averager()
    t0, micro, step, bad, oom, losses, cycle = time.time(), 0, 0, 0, 0, [], 0
    stop = None
    model.train()
    while stop is None:
        for img, ids in dl:
            masks, labels, keep = targets(ids, dev, semantic)
            if not keep:
                continue
            x = (img[keep].to(dev, non_blocking=True).float() - mean) / std
            if smoke and micro == 0:
                n = min(model.config.num_queries, dataset.max_instances)
                masks = [crowded_targets(n, *x.shape[-2:], dev) for _ in keep]
                labels = [torch.zeros(len(m), dtype=torch.long, device=dev) for m in masks]
            frac = min(step / cfg.smoke_steps, 1.0) if smoke else min((time.time() - t0) / budget, 1.0)
            try:
                with torch.autocast("cuda", dtype=torch.float16):
                    loss = model(pixel_values=x, mask_labels=masks, class_labels=labels).loss
                if not torch.isfinite(loss):
                    bad += 1
                    opt.zero_grad(set_to_none=True)
                    if bad > 50:
                        stop = "more than 50 non-finite losses"
                        break
                    continue
                scaler.scale(loss / accum).backward()
            except torch.cuda.OutOfMemoryError:
                if smoke:
                    raise
                oom += 1
                log(f"WARNING: CUDA OOM #{oom} at micro {micro} (batch skipped):\n" + traceback.format_exc(limit=1))
                del x, masks, labels
                loss = None
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                if oom > 20:
                    stop = "more than 20 CUDA OOMs"
                    break
                continue
            losses.append(loss.item())
            micro += 1
            if micro % accum == 0:
                if cycle < math.floor(frac * cfg.swa_cycles) < cfg.swa_cycles:
                    cycle = math.floor(frac * cfg.swa_cycles)
                    swa.add(model)
                    log(f"SWA snapshot {swa.n} at {frac:.2f} of the budget (end of a cycle, before the lr restart)")
                    if on_cycle is not None:
                        try:
                            on_cycle(f"cycle {cycle}", model)
                        except torch.cuda.OutOfMemoryError:
                            if smoke:
                                raise
                            log("WARNING: CUDA OOM in the quick val (skipped):\n" + traceback.format_exc(limit=1))
                            torch.cuda.empty_cache()
                        model.train()
                f = lr_factor(frac, step, cfg)
                for g in opt.param_groups:
                    g["lr"] = g["base_lr"] * f
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, cfg.clip_norm)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                step += 1
            if micro % 200 == 0 or smoke:
                el = time.time() - t0
                log(f"micro {micro} step {step} loss {np.mean(losses[-200:]):.3f} {el / micro:.2f} s/batch "
                    f"elapsed {el / 3600:.2f} h lr x{lr_factor(frac, step, cfg):.3f} "
                    f"mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GB bad {bad}")
            if smoke and step >= cfg.smoke_steps:
                frac_mem = torch.cuda.max_memory_allocated() / torch.cuda.get_device_properties(dev).total_memory
                assert frac_mem < mem_limit, (f"smoke peak GPU memory {frac_mem:.0%} of the card exceeds "
                                              f"{mem_limit:.0%}: the full run would risk OOM")
                stop = "smoke"
                break
            if not smoke and time.time() - t0 >= budget:
                stop = "budget"
                break
    if stop not in ("smoke", "budget"):
        log(f"WARNING: training stopped early: {stop}")
    opt.zero_grad(set_to_none=True)
    swa.add(model)
    swa.load_into(model)
    stats = {"micro": micro, "steps": step, "hours": (time.time() - t0) / 3600, "bad": bad, "oom": oom, "stop": stop.split(":")[0],
             "swa_snapshots": swa.n, "last_loss": float(np.mean(losses[-200:])) if losses else None,
             "max_mem_gb": torch.cuda.max_memory_allocated() / 2**30}
    log(f"training done: {stats}")
    model.eval()
    return stats
