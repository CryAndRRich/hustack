import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

ITERS = int(os.environ.get("ITERS", 2000))
PRE_ITERS = int(os.environ.get("PRE_ITERS", 4000))
BATCH = 8
CROP = 512
LR = 3e-4
WARMUP = 100
SCALE = 100.0
CELL = 16
DC_CELL, DC_CROP = 5, 160
DEVICE = torch.device("cuda")
MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        r = torchvision.models.resnet34(weights="IMAGENET1K_V1")
        self.body = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1, r.layer2, r.layer3)
        self.layer4, self.lat4 = r.layer4, nn.Conv2d(512, 256, 1)
        self.head = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(inplace=True),
                                  nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(64, 1, 1))

    def forward(self, x):
        f3 = self.body(x)
        f4 = F.interpolate(self.lat4(self.layer4(f3)), size=f3.shape[-2:], mode="bilinear", align_corners=False)
        return F.softplus(self.head(f3 + f4).float()).squeeze(1) / SCALE


def norm(x):
    return ((x - MEAN) / STD).contiguous(memory_format=torch.channels_last)


def sample(imgs, Y, idx, g, cell, size):
    pick = idx[torch.randint(len(idx), (BATCH,), generator=g)]
    gc = size // cell
    xs, ys = [], []
    for i in pick.tolist():
        r = int(torch.randint(Y.shape[1] - gc + 1, (1,), generator=g))
        c = int(torch.randint(Y.shape[2] - gc + 1, (1,), generator=g))
        x, y = imgs[i, :, r * cell:r * cell + size, c * cell:c * cell + size], Y[i, r:r + gc, c:c + gc]
        if torch.rand(1, generator=g) < 0.5:
            x, y = x.flip(-1), y.flip(-1)
        xs.append(x)
        ys.append(y)
    x = torch.stack(xs).float() / 255
    if size != CROP:
        x = F.interpolate(x, size=(CROP, CROP), mode="bilinear", align_corners=False)
    jitter = (torch.rand(BATCH, 2, 1, 1, 1, generator=g) * 0.4 + 0.8).to(DEVICE)
    m = x.mean((1, 2, 3), keepdim=True)
    x = ((x - m) * jitter[:, 0] + m * jitter[:, 1]).clamp(0, 1)
    return norm(x), torch.stack(ys), pick


def predict(net, imgs, idx):
    net.eval()
    out = []
    with torch.no_grad(), torch.autocast("cuda", torch.float16):
        for i in range(0, len(idx), 4):
            x = norm(imgs[torch.as_tensor(idx[i:i + 4])].float() / 255)
            out.append(((net(x) + net(x.flip(-1)).flip(-1)) / 2).float().cpu())
    net.train()
    return torch.cat(out).numpy()


def score(p, t):
    e = np.abs(p - t).sum((1, 2)) / t.sum((1, 2))
    return float(np.exp(-e.mean())), float(e.mean())


def fit(draw, cnt, iters, seed, deadline, init=None):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    net = Net()
    if init is not None:
        net.load_state_dict(torch.load(init, map_location="cpu", weights_only=True))
    net = net.to(DEVICE).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: min(1.0, (i + 1) / WARMUP) * 0.5 * (1 + math.cos(math.pi * min(i, iters) / iters)))
    scaler = torch.amp.GradScaler("cuda", init_scale=2 ** 12)
    t0, run = time.time(), 0.0
    for it in range(iters):
        if time.time() > deadline:
            print(f"  deadline hit at iter {it}/{iters}", flush=True)
            break
        x, y, pick = draw(g)
        with torch.autocast("cuda", torch.float16):
            p = net(x)
        loss = ((p - y).abs().sum((1, 2)) / cnt[pick]).mean()
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        run = 0.98 * run + 0.02 * loss.item() if it else loss.item()
        if it % 500 == 0 or it == iters - 1:
            print(f"  iter {it} loss {run:.4f} | {time.time() - t0:.0f}s", flush=True)
    return net


def pretrain(s, parts, deadline):
    imgs = torch.from_numpy(np.load(os.environ["DC_IMGS"])).permute(0, 3, 1, 2).to(DEVICE)
    maps = torch.from_numpy(np.load(os.environ["DC_MAPS"])).to(DEVICE)
    cnt, idx = maps.sum((1, 2)).clamp(min=1), torch.arange(len(maps))
    t0 = time.time()
    print(f"pretrain {s}: {len(maps)} Dense-Chicken frames, {PRE_ITERS} iters", flush=True)
    net = fit(lambda g: sample(imgs, maps, idx, g, DC_CELL, DC_CROP), cnt, PRE_ITERS, 7000 + s, deadline)
    tmp = parts / f"pre_{s}.tmp.pt"
    torch.save(net.state_dict(), tmp)
    os.replace(tmp, parts / f"pre_{s}.pt")
    print(f"pretrain {s}: done | {time.time() - t0:.0f}s", flush=True)


def run():
    parts, deadline = Path(os.environ["PARTS"]), float(os.environ["DEADLINE"])
    jobs = [j for j in os.environ["JOBS"].split(",") if j]
    if jobs and jobs[0].startswith("pre:"):
        for job in jobs:
            pretrain(int(job.split(":")[1]), parts, deadline)
        return
    data = np.load(os.environ["DATA"])
    n_pre = int(os.environ["N_PRE"])
    imgs = torch.from_numpy(np.load(os.environ["IMGS"])).permute(0, 3, 1, 2).to(DEVICE)
    Y, fold = torch.from_numpy(data["Y"]).to(DEVICE), data["fold"]
    cnt = Y.sum((1, 2))
    test_idx = np.arange(len(Y), len(imgs))
    for job in jobs:
        k, s = map(int, job.split(":"))
        a, b = np.flatnonzero(fold != k), np.flatnonzero(fold == k)
        t0 = time.time()
        print(f"job fold {k} seed {s}: train {len(a)} valid {len(b)} | init pre_{s % n_pre}", flush=True)
        idx = torch.as_tensor(a)
        net = fit(lambda g: sample(imgs, Y, idx, g, CELL, CROP), cnt, ITERS, 1000 * s + k, deadline,
                  init=parts / f"pre_{s % n_pre}.pt")
        oof, test = predict(net, imgs, b), predict(net, imgs, test_idx)
        S, e = score(oof, data["Y"][b])
        print(f"job fold {k} seed {s}: valid score {S:.4f} (mean e {e:.4f}) | {time.time() - t0:.0f}s", flush=True)
        for name, arr in (("oof", oof), ("test", test)):
            tmp = parts / f"{name}_{k}_{s}.tmp.npy"
            np.save(tmp, arr.astype(np.float32))
            os.replace(tmp, parts / f"{name}_{k}_{s}.npy")


if __name__ == "__main__":
    run()
