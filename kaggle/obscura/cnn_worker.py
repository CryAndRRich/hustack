import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

cfg = json.load(open(sys.argv[1]))
torch.set_num_threads(1)
if not torch.cuda.is_available():
    sys.exit("cnn worker: no GPU")
dev = torch.device("cuda")
d = np.load(cfg["data"])
Y, folds, n_tr = d["Y"], d["folds"], int(d["n_tr"])
fire = torch.tensor(Y[:, 4] > cfg["floor"] + 0.01, dtype=torch.float32, device=dev)
X = torch.from_numpy(np.load(cfg["cnn"])).to(dev)
mean = X.float().mean((0, 2, 3), keepdim=True)
std = X.float().std((0, 2, 3), keepdim=True) + 1e-3
X = ((X.float() - mean) / std).half()
CLOUDY = [0, 1, 3, 8, 9, 10]
N_HEAD = 4


def block(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                         nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class Net(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.f = nn.Sequential(block(c, 32), nn.MaxPool2d(2), block(32, 64), nn.MaxPool2d(2), block(64, 96),
                               nn.MaxPool2d(2), block(96, 128))
        self.drop, self.head = nn.Dropout(0.3), nn.Linear(256, N_HEAD)

    def forward(self, x):
        h = self.f(x)
        return self.head(self.drop(torch.cat([h.amax((2, 3)), h.mean((2, 3))], 1)))


def ext_inputs():
    xs, m, s = [], mean.cpu(), std.cpu()
    for p in cfg["ext_chips"]:
        a = np.load(p)["x"]
        r = a[:, :12].astype(np.float32) / 10000.0
        nocloud = ((a[:, 3] > 0) & ~np.isin(a[:, 12], CLOUDY)).astype(np.float32)
        tai = np.clip(a[:, 14].astype(np.float32) / 1000.0 - 1.0, -1, 3)
        x = np.stack([r[:, 3], r[:, 7], r[:, 8], r[:, 10], r[:, 11], nocloud, a[:, 13].astype(np.float32) / 10000.0, tai], 1)
        xs.append(((torch.from_numpy(np.clip(x, -1, 3)) - m) / s).half())
    return torch.cat(xs).to(dev)


def augment(x, zoom):
    if torch.rand(1) < 0.5:
        x = x.flip(3)
    if torch.rand(1) < 0.5:
        x = x.flip(2)
    x = torch.rot90(x, int(torch.randint(4, (1,))), (2, 3))
    if zoom and torch.rand(1) < 0.7:
        s = x.shape[-1]
        c = int(s * (0.5 + 0.5 * torch.rand(1)))
        i, j = int(torch.randint(s - c + 1, (1,))), int(torch.randint(s - c + 1, (1,)))
        x = F.interpolate(x[:, :, i:i + c, j:j + c].float(), size=s, mode="bilinear", align_corners=False).half()
    return x


def predict(net, src, idx):
    net.eval()
    out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        for i in range(0, len(idx), 256):
            x = src[idx[i:i + 256]]
            o = torch.stack([net(v).float() for v in (x, x.flip(3), x.flip(2), x.flip(2).flip(3))]).mean(0)
            out.append(torch.cat([torch.sigmoid(o[:, :1]), o[:, 1:]], 1))
    return torch.cat(out).cpu().numpy()


def train(net, src, idx, loss_fn, epochs, lr, bs, zoom):
    steps = epochs * ((len(idx) + bs - 1) // bs)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.2)
    scaler = torch.amp.GradScaler()
    net.train()
    loss = torch.zeros(())
    for _ in range(epochs):
        perm = idx[torch.randperm(len(idx), device=dev)]
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            with torch.autocast("cuda", dtype=torch.float16):
                loss = loss_fn(net(augment(src[b], zoom)).float(), b)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
    return loss.item()


t0 = time.time()
pre = None
if cfg["ext_chips"]:
    XE = ext_inputs()
    lab = np.load(cfg["ext_y"])
    YE = torch.tensor(lab["y"], dtype=torch.float32, device=dev)
    hold = torch.from_numpy(lab["hold"]).to(dev)

    def ext_loss(o, b):
        y = YE[b]
        loss = F.binary_cross_entropy_with_logits(o[:, 0], y[:, 0])
        for k in range(1, N_HEAD):
            m = ~torch.isnan(y[:, k])
            if m.any():
                loss = loss + F.smooth_l1_loss(o[m, k], y[m, k])
        return loss
    torch.manual_seed(7)
    pre = Net(X.shape[1]).to(dev).to(memory_format=torch.channels_last)
    last = train(pre, XE, torch.where(~hold)[0], ext_loss, cfg["ext_epochs"], 1e-3, 128, zoom=True)
    pv = predict(pre, XE, torch.where(hold)[0])
    yh = lab["y"][lab["hold"]]
    corr = [float(np.corrcoef(pv[:, 0], yh[:, 0])[0, 1])] + [
        float(np.corrcoef(pv[~np.isnan(yh[:, k]), k], yh[~np.isnan(yh[:, k]), k])[0, 1]) for k in range(1, N_HEAD)]
    print(f"cnn pretrain: {len(XE)} external chips, {time.time() - t0:.0f}s, last loss {last:.3f}; held-out facility "
          f"corr fire/aai/no2/co {np.round(corr, 3).tolist()}", flush=True)
    del XE, YE
    path = os.path.join(cfg["out"], "cnnext.npz")
    np.savez(path + ".tmp.npz", pred=predict(pre, X, torch.arange(len(X), device=dev)))
    os.replace(path + ".tmp.npz", path)

test_idx = torch.arange(n_tr, len(X), device=dev)
for r in range(len(folds)):
    for f in range(int(folds[r].max()) + 1):
        if time.time() > cfg["deadline"]:
            sys.exit(f"cnn worker: deadline before r{r} f{f}")
        tr_i = torch.from_numpy(np.where(folds[r] != f)[0]).to(dev)
        va_i = torch.from_numpy(np.where(folds[r] == f)[0]).to(dev)
        pv, pt = 0, 0
        for seed in range(cfg["seeds"]):
            torch.manual_seed(1000 * seed + 100 * r + f)
            net = Net(X.shape[1]).to(dev).to(memory_format=torch.channels_last)
            if pre is not None:
                net.load_state_dict(pre.state_dict())
            last = train(net, X, tr_i, lambda o, b: F.binary_cross_entropy_with_logits(o[:, 0], fire[b]),
                         cfg["epochs"], 1e-3 if pre is None else 3e-4, 64, zoom=False)
            pv = pv + predict(net, X, va_i)[:, 0] / cfg["seeds"]
            pt = pt + predict(net, X, test_idx)[:, 0] / cfg["seeds"]
        path = os.path.join(cfg["out"], f"cnn_r{r}_f{f}.npz")
        np.savez(path + ".tmp.npz", val=pv, test=pt)
        os.replace(path + ".tmp.npz", path)
        print(f"cnn r{r} f{f} done {time.time() - t0:.0f}s, last loss {last:.3f}", flush=True)
print("cnn worker finished", flush=True)
