import hashlib
import io
import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
from torch.nn.parallel import DistributedDataParallel as DDP
from PIL import Image

CLASSES = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
IMG = 64


CACHE = "fer_cache.npz"
CACHE_KEYS = {"train_ids", "train_labels", "X_tr", "md5_tr", "test_ids", "X_te", "md5_te"}


def find_input(name, root="/kaggle/input"):
    hits = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in ("train", "test"))
        if name in filenames:
            hits.append(Path(dirpath) / name)
    return sorted(hits, key=lambda p: (len(p.parts), str(p)))[0] if hits else None


def get_data_dir() -> Path:
    if Path("/kaggle/input").exists():
        for dirpath, dirnames, filenames in os.walk("/kaggle/input"):
            if "train.csv" in filenames and {"train", "test"} <= set(dirnames):
                with open(Path(dirpath) / "train.csv", encoding="utf-8") as f:
                    if f.readline().strip() == "id,label":
                        return Path(dirpath)
            dirnames[:] = [d for d in dirnames if d not in ("train", "test")]
        raise FileNotFoundError("no train.csv (id,label) with train/ and test/ under /kaggle/input")
    return Path(__file__).resolve().parent.parent / "data"


def _read(path):
    raw = path.read_bytes()
    with Image.open(io.BytesIO(raw)) as im:
        assert im.size == (IMG, IMG) and im.mode == "L", f"{path}: {im.size} {im.mode}"
        return np.asarray(im, dtype=np.uint8), hashlib.md5(raw).hexdigest()


def load_images(data_dir, threads=32):
    train = pd.read_csv(data_dir / "train.csv")
    assert train["id"].is_unique and set(train["label"].unique()) == set(range(7))
    cache = find_input(CACHE) if Path("/kaggle/input").exists() else None
    if cache is not None:
        d = np.load(cache)
        assert CACHE_KEYS <= set(d.files), f"{cache}: keys {d.files} — rebuild with src/build_cache.py"
        assert np.array_equal(d["train_ids"], train["id"]) and np.array_equal(d["train_labels"], train["label"]), \
            f"{cache} does not match train.csv — rebuild with src/build_cache.py"
        assert np.array_equal(d["test_ids"], np.arange(7000)) and d["X_te"].shape == (7000, IMG, IMG)
        print(f"images from cache {cache}")
        return (d["X_tr"], train["label"].to_numpy(np.int64), train["id"].to_numpy(), d["X_te"], d["test_ids"],
                d["md5_tr"].tolist(), d["md5_te"].tolist())
    test_ids = np.arange(7000)
    with ThreadPoolExecutor(threads) as ex:
        tr = list(ex.map(_read, [data_dir / "train" / f"{i}.jpg" for i in train["id"]]))
        te = list(ex.map(_read, [data_dir / "test" / f"{i}.jpg" for i in test_ids]))
    return (np.stack([a for a, _ in tr]), train["label"].to_numpy(np.int64), train["id"].to_numpy(),
            np.stack([a for a, _ in te]), test_ids, [h for _, h in tr], [h for _, h in te])


def near_pairs(X, skip, device, thr=0.98, chunk=4096):
    t = torch.from_numpy(X).to(device).float().unsqueeze(1)
    t = F.interpolate(t, size=16, mode="area").flatten(1)
    t = t - t.mean(1, keepdim=True)
    t = (t / (t.norm(dim=1, keepdim=True) + 1e-6)).half()
    t[torch.from_numpy(skip).to(device)] = 0
    out = []
    for s in range(0, len(t), chunk):
        c = t[s:s + chunk] @ t.T
        ii, jj = torch.nonzero(c > thr, as_tuple=True)
        keep = jj > ii + s
        out.append(torch.stack([ii[keep] + s, jj[keep]], 1).cpu().numpy())
        corr = c[ii[keep], jj[keep]].float().cpu().numpy()
        out[-1] = np.column_stack([out[-1], corr])
    del t
    torch.cuda.empty_cache()
    return np.concatenate(out) if out else np.zeros((0, 3))


def union_find_groups(n, pairs):
    parent = np.arange(n)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in pairs:
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return np.array([find(i) for i in range(n)])


FMAE_REPO = "forever208/FMAE-IAT"
FMAE_FILES = {"vit_small_patch16_224": "FMAE_ViT_small.pth", "vit_base_patch16_224": "FMAE_ViT_base.pth",
              "vit_large_patch16_224": "FMAE_ViT_large.pth"}
NEW_KEYS = {"fc_norm.weight", "fc_norm.bias", "head.weight", "head.bias"}


def build_model(cfg):
    import timm
    return timm.create_model(cfg["arch"], pretrained=False, num_classes=7, img_size=cfg["img_size"],
                             global_pool="avg", fc_norm=True, drop_path_rate=cfg["drop_path"])


def load_encoder(model, state):
    msg = model.load_state_dict(state, strict=False)
    missing = set(msg.missing_keys)
    assert missing <= NEW_KEYS, f"encoder weights missing from the FMAE checkpoint: {sorted(missing - NEW_KEYS)[:10]}"
    return msg


def init_head(model):
    nn.init.trunc_normal_(model.head.weight, std=2e-5)
    nn.init.zeros_(model.head.bias)


def param_groups(model, wd, layer_decay):
    n_layers = len(model.blocks) + 1
    groups = {}
    for name, prm in model.named_parameters():
        if name in ("cls_token", "pos_embed") or name.startswith("patch_embed"):
            layer = 0
        elif name.startswith("blocks."):
            layer = int(name.split(".")[1]) + 1
        else:
            layer = n_layers
        no_wd = prm.ndim == 1 or name in ("cls_token", "pos_embed")
        key = (layer, no_wd)
        if key not in groups:
            groups[key] = {"params": [], "weight_decay": 0.0 if no_wd else wd, "lr_scale": layer_decay ** (n_layers - layer)}
        groups[key]["params"].append(prm)
    return list(groups.values())


def augment(x, cfg):
    B, dev, crop, out = x.shape[0], x.device, cfg["crop"], cfg["img_size"]
    u = lambda: torch.rand(B, device=dev)
    on = lambda: (u() < cfg["p_aug"]).float()
    zoom = 1 + on() * (u() * 2 - 1) * cfg["scale"]
    ang = on() * (u() * 2 - 1) * math.radians(cfg["rot"])
    tx = on() * (u() * 2 - 1) * 2 * cfg["shift"]
    ty = on() * (u() * 2 - 1) * 2 * cfg["shift"]
    c = crop / IMG
    tx = tx + (u() * 2 - 1) * (1 - c)
    ty = ty + (u() * 2 - 1) * (1 - c)
    flip = torch.where(u() < 0.5, -1.0, 1.0)
    a = c / zoom
    cos, sin = torch.cos(ang), torch.sin(ang)
    theta = torch.stack([
        torch.stack([a * cos * flip, -a * sin, tx], 1),
        torch.stack([a * sin * flip, a * cos, ty], 1),
    ], 1)
    grid = F.affine_grid(theta, (B, 1, out, out), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)


def normalize(x, cfg):
    mean = torch.tensor(cfg["mean"], device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(cfg["std"], device=x.device).view(1, 3, 1, 1)
    return (x.expand(-1, 3, -1, -1) - mean) / std


def random_erase(x, cfg):
    B, _, H, W = x.shape
    dev = x.device
    area = (0.02 + torch.rand(B, device=dev) * 0.31) * H * W
    logr = math.log(0.3) + torch.rand(B, device=dev) * (math.log(3.3) - math.log(0.3))
    h = torch.sqrt(area * torch.exp(logr)).clamp(1, H).long()
    w = torch.sqrt(area / torch.exp(logr)).clamp(1, W).long()
    y0 = (torch.rand(B, device=dev) * (H - h + 1)).long()
    x0 = (torch.rand(B, device=dev) * (W - w + 1)).long()
    rows = torch.arange(H, device=dev)[None, :]
    cols = torch.arange(W, device=dev)[None, :]
    m = (((rows >= y0[:, None]) & (rows < (y0 + h)[:, None]))[:, :, None]
         & ((cols >= x0[:, None]) & (cols < (x0 + w)[:, None]))[:, None, :])
    m = m & (torch.rand(B, device=dev) < cfg["erase_p"])[:, None, None]
    return x.masked_fill(m[:, None], 0.0)


def _view(xb, oy, ox, cfg):
    crop = cfg["crop"]
    v = F.interpolate(xb[:, :, oy:oy + crop, ox:ox + crop], size=cfg["img_size"], mode="bilinear", align_corners=False)
    return normalize(v, cfg)


@torch.no_grad()
def predict(model, X, cfg, tta=True, bs=512):
    model.eval()
    dev, d = next(model.parameters()).device, IMG - cfg["crop"]
    offs = [(0, 0), (0, d), (d, 0), (d, d), (d // 2, d // 2)] if tta else [(d // 2, d // 2)]
    out = []
    for s in range(0, len(X), bs):
        xb = torch.from_numpy(X[s:s + bs]).to(dev).float().unsqueeze(1) / 255.0
        p = 0
        for oy, ox in offs:
            v = _view(xb, oy, ox, cfg)
            for view in ((v, v.flip(3)) if tta else (v,)):
                with torch.autocast("cuda", dtype=torch.float16):
                    p = p + F.softmax(model(view).float(), 1)
        out.append((p / (2 * len(offs) if tta else 1)).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 7), np.float32)


def predict_sharded(model, X, cfg, tta, rank, world):
    part = predict(model, X[rank::world], cfg, tta)
    if world == 1:
        return part
    parts = [None] * world
    dist.all_gather_object(parts, part)
    out = np.zeros((len(X), 7), np.float32)
    for r, q in enumerate(parts):
        out[r::world] = q
    return out


def train_model(cfg, X_tr, y_tr, X_va, y_va, X_te, seed, log=print, rank=0, world=1):
    dev = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    Xg = torch.from_numpy(X_tr).to(dev).unsqueeze(1)
    yg = torch.from_numpy(y_tr).to(dev)
    net = build_model(cfg)
    load_encoder(net, torch.load(cfg["init"], map_location="cpu", weights_only=True))
    init_head(net)
    net = net.to(dev)
    if world > 1:
        model = DDP(net, device_ids=[dev.index], broadcast_buffers=False, gradient_as_bucket_view=True)
        model.register_comm_hook(None, default_hooks.fp16_compress_hook)
    else:
        model = net
    torch.manual_seed(seed + 1000 * rank)
    groups = param_groups(net, cfg["wd"], cfg["layer_decay"])
    for g in groups:
        g["lr"] = cfg["lr"] * g["lr_scale"]
    opt = torch.optim.AdamW(groups, lr=cfg["lr"], betas=(0.9, 0.999))
    bs = cfg["bs"]
    steps = len(Xg) // bs
    total, warm = cfg["epochs"] * steps, max(1, cfg["warmup_epochs"] * steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    scaler = torch.amp.GradScaler("cuda", init_scale=2.0 ** 10)
    crit = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    gen = torch.Generator(device=dev)
    hist = []
    for ep in range(cfg["epochs"]):
        t0 = time.time()
        model.train()
        gen.manual_seed(seed * 1000 + ep)
        perm = torch.randperm(len(Xg), device=dev, generator=gen)
        stats = torch.zeros(3, device=dev)
        for b in range(steps):
            idx = perm[b * bs:(b + 1) * bs][rank::world]
            xb = random_erase(normalize(augment(Xg[idx].float() / 255.0, cfg), cfg), cfg)
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(xb)
                loss = crit(logits, yg[idx])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            stats[0] += loss.detach() * len(idx)
            stats[1] += (logits.argmax(1) == yg[idx]).sum()
            stats[2] += len(idx)
        if world > 1:
            dist.all_reduce(stats)
        rec = {"epoch": ep + 1, "loss": stats[0].item() / stats[2].item(), "train_acc": stats[1].item() / stats[2].item(),
               "lr": max(g["lr"] for g in opt.param_groups), "sec": time.time() - t0}
        msg = (f"    ep {ep + 1:3d}  loss {rec['loss']:.4f}  train_acc {rec['train_acc']:.4f}  lr(top) {rec['lr']:.2e}  "
               f"{rec['sec']:.0f}s")
        if cfg["eval_every"] and ((ep + 1) % cfg["eval_every"] == 0 or ep + 1 == cfg["epochs"]):
            pv = predict_sharded(net, X_va, cfg, False, rank, world)
            rec["val_acc_centre"] = float((pv.argmax(1) == y_va).mean())
            msg += f"  val_acc(centre) {rec['val_acc_centre']:.4f}"
        log(msg)
        hist.append(rec)
    t0 = time.time()
    va_probs = predict_sharded(net, X_va, cfg, cfg["tta"], rank, world)
    te_probs = predict_sharded(net, X_te, cfg, cfg["tta"], rank, world)
    log(f"  prediction ({'ten-crop TTA' if cfg['tta'] else 'centre crop'}, val + test) in {time.time() - t0:.0f}s")
    if rank != 0:
        return None
    state = {k: v.detach().half().cpu() for k, v in net.state_dict().items()}
    del model, net, Xg, yg, opt
    torch.cuda.empty_cache()
    return va_probs, te_probs, hist, state
