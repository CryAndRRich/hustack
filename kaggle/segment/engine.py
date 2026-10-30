import json
import math
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
from torch.nn.parallel import DistributedDataParallel as DDP

from model import Sapiens2Seg, load_sapiens2

SIZE = 320
N_CLS = 20
IGNORE = 255
PAD = (124, 116, 104)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
SWAP = list(range(14)) + [15, 14, 17, 16, 19, 18]
NAMES = ("background", "hat", "hair", "glove", "sunglasses", "upper", "dress", "coat", "socks", "pants", "torso",
         "scarf", "skirt", "face", "l-arm", "r-arm", "l-leg", "r-leg", "l-shoe", "r-shoe")


def window(cfg):
    w = SIZE * cfg["img"] // cfg["frame"]
    assert w * cfg["frame"] == SIZE * cfg["img"] and 2 * w == cfg["img"], cfg
    return w


def _sample(x, theta, side):
    dev = x.device
    pad = torch.tensor(PAD, device=dev, dtype=torch.float32).view(1, 3, 1, 1)
    grid = F.affine_grid(theta, (len(x), 3, side, side), align_corners=False)
    xa = F.grid_sample(x - pad, grid, mode="bilinear", padding_mode="zeros", align_corners=False) + pad
    mean = torch.tensor(MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=dev).view(1, 3, 1, 1)
    return (xa / 255 - mean) / std


def views_train(x, y, cfg):
    B, dev, W = len(x), x.device, window(cfg)
    x = x.permute(0, 3, 1, 2).float()

    def u():
        return torch.rand(B, device=dev)

    on = (u() < cfg["p_jitter"]).float().view(B, 1, 1, 1)

    def jit():
        return 1 + on * (u().view(B, 1, 1, 1) * 2 - 1) * cfg["jitter"]

    gray = x.mean(1, keepdim=True)
    x = gray + (x - gray) * jit()
    m = x.mean((1, 2, 3), keepdim=True)
    x = ((m + (x - m) * jit()) * jit()).clamp_(0, 255)
    lo, hi = map(math.log, cfg["zoom"])
    a = (W / SIZE) / torch.exp(lo + u() * (hi - lo))
    ang = (u() * 2 - 1) * math.radians(cfg["rot"]) * (u() < cfg["p_rot"]).float()
    flip = torch.where(u() < 0.5, -1.0, 1.0)
    fg = ((y > 0) & (y != IGNORE)).flatten(1).float()
    pick = torch.multinomial(fg + 1e-6, 1).squeeze(1)
    use = u() < cfg["p_fg"]
    cx = torch.where(use, ((pick % SIZE).float() + 0.5) / SIZE * 2 - 1, u() * 2 - 1)
    cy = torch.where(use, ((pick // SIZE).float() + 0.5) / SIZE * 2 - 1, u() * 2 - 1)
    lim = (1 - a).clamp(min=0)
    cx, cy = torch.maximum(torch.minimum(cx, lim), -lim), torch.maximum(torch.minimum(cy, lim), -lim)
    cos, sin = ang.cos() * a, ang.sin() * a
    theta = torch.stack([torch.stack([cos * flip, -sin, cx], 1), torch.stack([sin * flip, cos, cy], 1)], 1)
    xa = _sample(x, theta, cfg["img"])
    grid = F.affine_grid(theta, (B, 1, W, W), align_corners=False)
    ya = F.grid_sample(y.unsqueeze(1).float() + 1, grid, mode="nearest", padding_mode="zeros",
                       align_corners=False).squeeze(1).long() - 1
    ya = torch.where(ya < 0, IGNORE, ya)
    lut = torch.arange(256, device=dev)
    lut[:N_CLS] = torch.tensor(SWAP, device=dev)
    return xa, torch.where((flip < 0).view(B, 1, 1), lut[ya], ya)


def window_starts(w):
    return 0, SIZE - w


def window_views(x, w, img):
    B, dev = len(x), x.device
    x = x.permute(0, 3, 1, 2).float()
    a = w / SIZE
    centres = [(s + w / 2) / SIZE * 2 - 1 for s in window_starts(w)]
    theta = torch.tensor([[[a, 0, cx], [0, a, cy]] for cy in centres for cx in centres], device=dev,
                         dtype=torch.float32)
    return _sample(x.repeat_interleave(4, 0), theta.repeat(B, 1, 1), img)


def logits(net, xa, chunk=None):
    with torch.autocast("cuda", dtype=torch.float16):
        if chunk is None:
            return net(xa).float()
        return torch.cat([net(c).float() for c in xa.split(chunk)])


def window_probs(net, x, cfg, flip, w=None, chunk=16):
    B, w = len(x), w or window(cfg)
    xa = window_views(x, w, cfg["img"])
    p = logits(net, xa, chunk).softmax(1)
    if flip:
        p = (p + logits(net, xa.flip(3), chunk).softmax(1).flip(3)[:, SWAP]) / 2
    if p.shape[-1] != w:
        p = F.interpolate(p, size=(w, w), mode="bilinear", align_corners=False, antialias=True)
    p = p.view(B, 4, N_CLS, w, w)
    out = torch.zeros(B, N_CLS, SIZE, SIZE, device=x.device)
    cnt = torch.zeros(1, 1, SIZE, SIZE, device=x.device)
    starts = window_starts(w)
    for k, (sy, sx) in enumerate((sy, sx) for sy in starts for sx in starts):
        out[..., sy:sy + w, sx:sx + w] += p[:, k]
        cnt[..., sy:sy + w, sx:sx + w] += 1
    return out / cnt


def valid_from_image(x):
    pad = (x == torch.tensor(PAD, device=x.device, dtype=x.dtype)).all(-1)
    return ~(pad.all(2, keepdim=True) | pad.all(1, keepdim=True))


def lovasz(logits, y):
    B, C = logits.shape[:2]
    p = logits.softmax(1).flatten(2)
    y = y.flatten(1)
    valid = (y != IGNORE).unsqueeze(1)
    fg = (y.unsqueeze(1) == torch.arange(C, device=y.device).view(1, C, 1)) & valid
    err = (fg.float() - p).abs() * valid
    err, order = err.sort(2, descending=True)
    fg = fg.float().gather(2, order)
    gts = fg.sum(2, keepdim=True)
    jac = 1 - (gts - fg.cumsum(2)) / (gts + (1 - fg).cumsum(2))
    jac = torch.cat([jac[..., :1], jac[..., 1:] - jac[..., :-1]], 2)
    loss = (err * jac).sum(2)
    present = gts.squeeze(2) > 0
    return ((loss * present).sum(1) / present.sum(1).clamp(min=1)).mean()


def confusion(pred, gt):
    B = len(pred)
    valid = gt != IGNORE
    idx = (torch.arange(B, device=pred.device).view(B, 1, 1) * N_CLS + gt.long().clamp(max=N_CLS - 1)) * N_CLS + pred
    return torch.bincount(idx[valid], minlength=B * N_CLS * N_CLS).view(B, N_CLS, N_CLS)


def miou(conf):
    conf = conf.double()
    inter = conf.diagonal(dim1=1, dim2=2)
    a_gt, a_pr = conf.sum(2), conf.sum(1)
    union = a_gt + a_pr - inter
    iou = inter / union.clamp(min=1)
    pu, pg = union > 0, a_gt > 0
    return (iou * pu).sum(1) / pu.sum(1).clamp(min=1), (iou * pg).sum(1) / pg.sum(1).clamp(min=1)


def param_groups(net, cfg):
    n = len(net.backbone.blocks) + 1
    groups = {}
    for name, p in net.named_parameters():
        if not p.requires_grad:
            continue
        layer = int(name.split(".")[2]) + 1 if name.startswith("backbone.blocks.") else n
        scale, no_wd = cfg["layer_decay"] ** (n - layer), p.ndim == 1
        g = groups.setdefault((scale, no_wd), {"params": [], "base_lr": cfg["lr"] * scale, "lr": cfg["lr"] * scale,
                                              "weight_decay": 0.0 if no_wd else cfg["wd"]})
        g["params"].append(p)
    return list(groups.values())


@torch.no_grad()
def evaluate(net, X, Y, cfg, rows, rank, world, bs=4):
    net.eval()
    dev = next(net.parameters()).device
    s = torch.zeros(4 + 2 * N_CLS, device=dev, dtype=torch.float64)
    idx = rows[rank::world]
    for c in range(0, len(idx), bs):
        j = idx[c:c + bs]
        p = window_probs(net, torch.from_numpy(X[j]).to(dev), cfg, False)
        cm = confusion(p.argmax(1), torch.from_numpy(Y[j]).to(dev)).double()
        u, g = miou(cm)
        inter, a_gt, a_pr = cm.diagonal(dim1=1, dim2=2).sum(0), cm.sum((0, 2)), cm.sum((0, 1))
        s += torch.cat([torch.stack([u.sum(), g.sum(), u.new_tensor(float(len(j))),
                                     (~torch.isfinite(p)).sum().double()]), inter, a_gt + a_pr - inter])
    if world > 1:
        dist.all_reduce(s)
    iou = (s[4:4 + N_CLS] / s[4 + N_CLS:].clamp(min=1)).tolist()
    return (s[0] / s[2]).item(), (s[1] / s[2]).item(), iou, int(s[3].item())


@torch.no_grad()
def predict(net, X, cfg, rank, world, out, split, bs=4):
    net.eval()
    dev = next(net.parameters()).device
    arms = {"ss": [window(cfg)], "ms": cfg["tta"]}
    idx = np.arange(rank, len(X), world)
    top3 = {a: np.lib.format.open_memmap(out / f"{split}_{a}_top3_{rank}.npy", "w+", np.uint8, (len(idx), 3, SIZE, SIZE))
            for a in arms}
    stats = {a: np.zeros((len(idx), N_CLS, 2), np.float32) for a in arms}
    for c in range(0, len(idx), bs):
        j = idx[c:c + bs]
        x = torch.from_numpy(X[j]).to(dev)
        probs = {w: window_probs(net, x, cfg, True, w) for w in cfg["tta"]}
        valid = valid_from_image(x)
        for a, ws in arms.items():
            val, ids = (sum(probs[w] for w in ws) / len(ws)).topk(3, 1)
            b = (torch.arange(len(j), device=dev).view(-1, 1, 1) * N_CLS + ids[:, 0])[valid]
            area = torch.bincount(b, minlength=len(j) * N_CLS).view(-1, N_CLS).float()
            conf = torch.bincount(b, weights=val[:, 0][valid], minlength=len(j) * N_CLS).view(-1, N_CLS)
            stats[a][c:c + len(j)] = torch.stack([area, conf / area.clamp(min=1)], 2).cpu().numpy()
            top3[a][c:c + len(j)] = ids.to(torch.uint8).cpu().numpy()
    for a in arms:
        top3[a].flush()
        np.save(out / f"{split}_{a}_stats_{rank}.npy", stats[a])


def train(cfg, X, Y, Xv, Yv, Xt, ckpt, out, rank, world, log):
    dev = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(cfg["seed"] + rank)
    net = Sapiens2Seg(cfg["img"], N_CLS, cfg["freeze"], cfg["ckpt_blocks"])
    load_sapiens2(net, ckpt)
    net.to(dev)
    model = net
    if world > 1:
        model = DDP(net, device_ids=[dev.index], broadcast_buffers=False, gradient_as_bucket_view=True,
                    static_graph=True)
        model.register_comm_hook(None, default_hooks.fp16_compress_hook)
    opt = torch.optim.AdamW(param_groups(net, cfg), betas=(0.9, 0.999), fused=True)
    trainable = [p for p in net.parameters() if p.requires_grad]
    log(f"  {sum(p.numel() for p in net.parameters()) / 1e6:.0f}M params, {sum(p.numel() for p in trainable) / 1e6:.0f}M "
        f"trainable (first {cfg['freeze']} blocks frozen, {cfg['ckpt_blocks']} checkpointed); crops {cfg['img']} of the "
        f"{cfg['frame']} frame = {window(cfg)} source px")
    sub = np.arange(0, len(Xv), cfg["eval_every"])
    t0 = time.time()
    v0 = evaluate(net, Xv, Yv, cfg, sub, rank, world)
    eval_s = time.time() - t0
    assert window(cfg) in cfg["tta"], cfg["tta"]
    infer_s = eval_s * (len(Xv) + len(Xt)) * 2 * len(cfg["tta"]) / len(sub)
    log(f"    step 0: val[::{cfg['eval_every']}] union {v0[0]:.4f} gt {v0[1]:.4f}, non-finite probs {v0[3]} "
        f"({eval_s:.0f}s; final inference, {len(cfg['tta'])} scales x flip, est. {infer_s / 60:.0f} min), "
        f"peak {torch.cuda.max_memory_allocated(dev) / 2 ** 30:.1f} GiB")
    log("    step 0 class IoU: " + ", ".join(f"{n} {v:.2f}" for n, v in zip(NAMES, v0[2])))
    torch.cuda.reset_peak_memory_stats(dev)
    scaler = torch.amp.GradScaler("cuda", init_scale=2.0 ** 12)
    bs, per_epoch, warm, steps = cfg["bs"], len(X) // cfg["bs"], cfg["warmup"], cfg.get("steps")
    t_start = time.time()
    budget = cfg["deadline"] - cfg["post_min"] * 60 - cfg["infer_margin"] * infer_s - t_start
    log(f"    training budget {budget / 60:.0f} min (deadline minus {cfg['post_min']} min post-processing and "
        f"{cfg['infer_margin']}x the inference estimate); LR decays with wall time")
    prog = torch.zeros((), device=dev)
    step, ep, t_probe, stop, p = 0, 0, None, False, 0.0
    hist = [{"epoch": 0, "val_union": v0[0], "val_gt": v0[1], "class_iou": v0[2], "nonfinite": v0[3],
             "eval_s": eval_s, "budget_min": budget / 60}]
    while not stop:
        model.train()
        perm = np.random.default_rng(cfg["seed"] + ep).permutation(len(X))
        stats = torch.zeros(3, device=dev)
        t0, done = time.time(), 0
        for b in range(per_epoch):
            if steps is not None:
                prog.fill_(step / steps)
            elif step % 10 == 0:
                prog.fill_((time.time() - t_start) / budget)
                if world > 1:
                    dist.all_reduce(prog, op=dist.ReduceOp.MAX)
            if steps is not None or step % 10 == 0:
                p = prog.item()
            if p >= 1:
                stop = True
                break
            j = np.sort(perm[b * bs:(b + 1) * bs][rank::world])
            xa, ya = views_train(torch.from_numpy(X[j]).to(dev), torch.from_numpy(Y[j]).to(dev), cfg)
            lg = logits(model, xa)
            ce, lv = F.cross_entropy(lg, ya, ignore_index=IGNORE), lovasz(lg, ya)
            f = min(1.0, (step + 1) / warm) * (1 - p)
            for g in opt.param_groups:
                g["lr"] = g["base_lr"] * f
            opt.zero_grad(set_to_none=True)
            scaler.scale(ce + cfg["lovasz"] * lv).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(trainable, cfg["clip"])
            scaler.step(opt)
            scaler.update()
            stats += torch.stack([ce.detach(), lv.detach(), torch.ones((), device=dev)])
            step, done = step + 1, done + 1
            if steps is None and step in cfg["probe"]:
                torch.cuda.synchronize()
                if step == cfg["probe"][0]:
                    t_probe = time.time()
                else:
                    rate = bs * (cfg["probe"][1] - cfg["probe"][0]) / (time.time() - t_probe)
                    log(f"    {rate:.1f} img/s, peak {torch.cuda.max_memory_allocated(dev) / 2 ** 30:.1f} GiB -> about "
                        f"{rate * budget / len(X):.1f} epochs in the budget")
        if world > 1:
            dist.all_reduce(stats)
        rec = {"epoch": ep + 1, "steps": step, "ce": (stats[0] / stats[2].clamp(min=1)).item(),
               "lovasz": (stats[1] / stats[2].clamp(min=1)).item(), "img_s": done * bs / (time.time() - t0),
               "peak_gib": torch.cuda.max_memory_allocated(dev) / 2 ** 30}
        if done == per_epoch:
            rec["val_union"], rec["val_gt"] = evaluate(net, Xv, Yv, cfg, sub, rank, world)[:2]
        rec["min"] = (time.time() - t_start) / 60
        hist.append(rec)
        log("    " + "  ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in rec.items()))
        ep += 1
    if rank == 0:
        torch.save({k: v.half() for k, v in net.state_dict().items()}, out / "model.pt")
    t0 = time.time()
    predict(net, Xv, cfg, rank, world, out, "val")
    predict(net, Xt, cfg, rank, world, out, "test")
    infer = time.time() - t0
    log(f"  val + test multi-scale flip-TTA window inference in {infer / 60:.1f} min (estimated {infer_s / 60:.1f})")
    if rank == 0:
        (out / "hist.json").write_text(json.dumps({"hist": hist, "world": world, "infer_s": infer}))
