import math
import time

import numpy as np
import timm
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
from torch.nn.parallel import DistributedDataParallel as DDP


def build(cfg):
    net = timm.create_model(cfg["arch"], pretrained=True, num_classes=1, drop_path_rate=cfg["drop_path"])
    net.set_grad_checkpointing(cfg["grad_ckpt"])
    torch.nn.init.zeros_(net.head.weight)
    torch.nn.init.zeros_(net.head.bias)
    return net


def param_groups(net, cfg):
    n = len(net.blocks) + 1
    groups = {}
    for name, p in net.named_parameters():
        if name.startswith(("cls_token", "pos_embed", "patch_embed", "norm_pre")):
            layer = 0
        elif name.startswith("blocks."):
            layer = int(name.split(".")[1]) + 1
        else:
            layer = n
        scale = cfg["layer_decay"] ** (n - layer) * (cfg["head_lr_mult"] if name.startswith("head.") else 1.0)
        no_wd = p.ndim == 1 or name in ("cls_token", "pos_embed")
        g = groups.setdefault((scale, no_wd), {"params": [], "lr": cfg["lr"] * scale,
                                              "weight_decay": 0.0 if no_wd else cfg["wd"]})
        g["params"].append(p)
    return list(groups.values())


def views(x, cfg, train):
    B, dev = len(x), x.device
    x = x.float().div_(255).unsqueeze(1)
    if train:
        def u():
            return torch.rand(B, device=dev)
        area = cfg["scale"][0] + u() * (cfg["scale"][1] - cfg["scale"][0])
        logr = math.log(cfg["ratio"][0]) + u() * (math.log(cfg["ratio"][1]) - math.log(cfg["ratio"][0]))
        sx, sy = torch.sqrt(area * logr.exp()).clamp(max=1), torch.sqrt(area / logr.exp()).clamp(max=1)
        tx, ty = (u() * 2 - 1) * (1 - sx), (u() * 2 - 1) * (1 - sy)
        ang = (u() * 2 - 1) * math.radians(cfg["rot"])
        flip = torch.where(u() < 0.5, -1.0, 1.0)
        cos, sin = ang.cos(), ang.sin()
        theta = torch.stack([torch.stack([sx * cos * flip, -sy * sin, tx], 1),
                             torch.stack([sx * sin * flip, sy * cos, ty], 1)], 1)
    else:
        theta = torch.eye(2, 3, device=dev).repeat(B, 1, 1)
    grid = F.affine_grid(theta, (B, 1, cfg["img"], cfg["img"]), align_corners=False)
    x = F.grid_sample(x, grid, mode="bicubic", padding_mode="border", align_corners=False).clamp_(0, 1)
    if train:
        on = (u() < cfg["p_jitter"]).float().view(B, 1, 1, 1)
        c = 1 + on * (u().view(B, 1, 1, 1) * 2 - 1) * cfg["jitter"]
        b = on * (u().view(B, 1, 1, 1) * 2 - 1) * cfg["jitter"] / 2
        m = x.mean((1, 2, 3), keepdim=True)
        x = ((x - m) * c + m + b).clamp_(0, 1)
    mean = torch.tensor(cfg["mean"], device=dev).view(1, 3, 1, 1)
    std = torch.tensor(cfg["std"], device=dev).view(1, 3, 1, 1)
    return (x - mean) / std


@torch.no_grad()
def predict(net, X, cfg, flip, bs=128):
    net.eval()
    dev = next(net.parameters()).device
    out = []
    for s in range(0, len(X), bs):
        v = views(torch.from_numpy(np.ascontiguousarray(X[s:s + bs])).to(dev), cfg, False)
        with torch.autocast("cuda", dtype=torch.float16):
            p = net(v).float().squeeze(1)
            if flip:
                p = (p + net(v.flip(3)).float().squeeze(1)) / 2
        out.append(p.cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, np.float32)


def predict_sharded(net, X, cfg, flip, rank, world):
    part = predict(net, X[rank::world], cfg, flip)
    if world == 1:
        return part
    parts = [None] * world
    dist.all_gather_object(parts, part)
    out = np.zeros(len(X), np.float32)
    for r, q in enumerate(parts):
        out[r::world] = q
    return out


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def train(cfg, X_tr, y_tr, X_va, y_va, mon, X_te, rank, world, log):
    dev = torch.device("cuda", torch.cuda.current_device())
    ym, ys = float(y_tr.mean()), float(y_tr.std())
    if world > 1 and rank > 0:
        dist.barrier()
    net = build(cfg).to(dev)
    if world > 1 and rank == 0:
        dist.barrier()
    model = net
    if world > 1:
        model = DDP(net, device_ids=[dev.index], broadcast_buffers=False, gradient_as_bucket_view=True,
                    static_graph=True)
        model.register_comm_hook(None, default_hooks.fp16_compress_hook)
    torch.manual_seed(cfg["seed"] + 1000 * rank)
    opt = torch.optim.AdamW(param_groups(net, cfg), betas=(0.9, 0.999), fused=True)
    Xg = torch.from_numpy(X_tr).to(dev)
    yg = torch.from_numpy(((y_tr - ym) / ys).astype(np.float32)).to(dev)
    bs, steps = cfg["bs"], len(X_tr) // cfg["bs"]
    total, warm = cfg["epochs"] * steps, max(1, int(cfg["warmup"] * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    scaler = torch.amp.GradScaler("cuda", init_scale=2.0 ** 10)
    gen = torch.Generator(device=dev)
    X_mon, y_mon = X_va[mon], y_va[mon]
    best, best_state, best_epoch, hist = math.inf, None, 0, []
    for ep in range(cfg["epochs"]):
        t0 = time.time()
        model.train()
        gen.manual_seed(cfg["seed"] * 1000 + ep)
        perm = torch.randperm(len(Xg), device=dev, generator=gen)
        stats = torch.zeros(2, device=dev)
        for b in range(steps):
            idx = perm[b * bs:(b + 1) * bs][rank::world]
            xb = views(Xg[idx], cfg, True)
            with torch.autocast("cuda", dtype=torch.float16):
                pred = model(xb).float().squeeze(1)
            loss = F.mse_loss(pred, yg[idx])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            stats += torch.stack([loss.detach() * len(idx), torch.tensor(float(len(idx)), device=dev)])
            if ep == 0 and b == min(30, steps - 1):
                log(f"    first {b + 1} steps: {(b + 1) * bs / (time.time() - t0):.0f} img/s, peak memory "
                    f"{torch.cuda.max_memory_allocated(dev) / 2 ** 30:.1f} GiB, est. {steps * (time.time() - t0) / (b + 1) / 60:.0f} min/epoch")
        if world > 1:
            dist.all_reduce(stats)
        t_train = time.time() - t0
        p = np.clip(predict_sharded(net, X_mon, cfg, False, rank, world) * ys + ym, *cfg["age_range"])
        r = rmse(y_mon, p)
        if r < best:
            best, best_epoch = r, ep + 1
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        rec = {"epoch": ep + 1, "train_rmse": math.sqrt(stats[0].item() / stats[1].item()) * ys, "monitor_rmse": r,
               "lr_top": max(g["lr"] for g in opt.param_groups), "train_s": t_train, "img_s": steps * bs / t_train,
               "s": time.time() - t0, "peak_gib": torch.cuda.max_memory_allocated(dev) / 2 ** 30}
        hist.append(rec)
        log(f"    ep {ep + 1}  train rmse {rec['train_rmse']:.3f}  monitor rmse {r:.4f}  {rec['img_s']:.0f} img/s  "
            f"{rec['s']:.0f}s  peak {rec['peak_gib']:.1f} GiB{'  *' if best_epoch == ep + 1 else ''}")
    net.load_state_dict(best_state)
    t0 = time.time()
    pv = predict_sharded(net, X_va, cfg, True, rank, world) * ys + ym
    pt = predict_sharded(net, X_te, cfg, True, rank, world) * ys + ym
    log(f"  best epoch {best_epoch} (monitor {best:.4f}); val + judge with mirror TTA in {time.time() - t0:.0f}s")
    return pv, pt, hist, best_epoch
