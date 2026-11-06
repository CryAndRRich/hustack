import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

FREQS = np.round(np.arange(8.0, 10.0001, 0.1), 1)
COLS = [f"A_{f:.1f}" for f in FREQS]
G0, DG, NG = 6.5, 0.1, 59
SHIFT = 0.1
PX = 0.0155
NFOLD, SEEDS = 5, 6
JOBS = [(f, s) for s in range(SEEDS) for f in [*range(NFOLD), -1]]
EPOCHS, BATCH, LR, WD = 400, 32, 2e-3, 1e-4
LOG_K = 0.06
U_BATCH, LAMBDA, RAMP = 16, 1.0, 0.2


def features(phi):
    return torch.stack([(phi / PX).clamp(-3, 3) / 3, (phi > 0).float()], 1)


def rescale(phi, k):
    theta = torch.zeros(len(phi), 2, 3, device=phi.device)
    theta[:, 0, 0] = theta[:, 1, 1] = 1 / k
    grid = F.affine_grid(theta, (len(phi), 1, *phi.shape[1:]), align_corners=False)
    return F.grid_sample(phi[:, None], grid, padding_mode="border", align_corners=False)[:, 0]


def block(i, o):
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1, padding_mode="circular", bias=False), nn.BatchNorm2d(o), nn.GELU(),
        nn.Conv2d(o, o, 3, padding=1, padding_mode="circular", bias=False), nn.BatchNorm2d(o), nn.GELU())


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        ch = [2, 32, 64, 128, 128]
        self.convs = nn.ModuleList(block(a, b) for a, b in zip(ch, ch[1:]))
        self.head = nn.Sequential(nn.Linear(ch[-1] + 1, 256), nn.GELU(), nn.Linear(256, NG + 2))
        self.a = nn.Parameter(torch.tensor(1.29))
        self.b = nn.Parameter(torch.tensor(0.15))
        self.register_buffer("freqs", torch.tensor(FREQS, dtype=torch.float32))

    def forward(self, x, s, t):
        with torch.autocast("cuda", dtype=torch.float16):
            for i, conv in enumerate(self.convs):
                x = conv(x)
                if i < len(self.convs) - 1:
                    x = F.avg_pool2d(x, 2)
        h = self.head(torch.cat([x.float().mean((2, 3)), (t[:, None] - 0.86) / 0.04], 1))
        canon, da, shift = torch.sigmoid(h[:, :NG]), 0.3 * torch.tanh(h[:, NG]), SHIFT * torch.tanh(h[:, NG + 1])
        r = torch.exp((self.a + da) * torch.log(s / 1.2) + self.b * (t - 0.86) + shift)
        u = ((self.freqs[None] * r[:, None] - G0) / DG).clamp(0, NG - 1 - 1e-4)
        i0 = u.floor().long()
        w = u - i0
        return canon.gather(1, i0) * (1 - w) + canon.gather(1, i0 + 1) * w


def augment(phi, s):
    n, dev = len(phi), phi.device
    k = torch.where(torch.rand(n, device=dev) < 0.5, torch.exp((2 * torch.rand(n, device=dev) - 1) * LOG_K), torch.ones(n, device=dev))
    phi = rescale(phi, k)
    sh = torch.randint(0, phi.shape[-1], (n, 2)).tolist()
    phi = torch.stack([torch.roll(p, (a, b), (0, 1)) for p, (a, b) in zip(phi, sh)])
    for dim in (1, 2):
        flip = torch.rand(n, device=dev) < 0.5
        phi = torch.where(flip[:, None, None], phi.flip(dim), phi)
    return features(phi), s / k


def arm(seed):
    return seed % 2


def fit(X, S, T, Y, U, Su, Tu, seed):
    m = U_BATCH * arm(seed)
    torch.manual_seed(seed)
    net = Net().to(X.device)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    steps = EPOCHS * -(-len(X) // BATCH)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler()
    net.train()
    step = 0
    for _ in range(EPOCHS):
        for idx in torch.randperm(len(X), device=X.device).split(BATCH):
            if len(idx) < 2:
                continue
            n, iu = len(idx), torch.randint(0, len(U), (m,), device=X.device)
            views, ts = [augment(X[idx], S[idx])], [T[idx]]
            if m:
                views += [augment(U[iu], Su[iu]), augment(U[iu], Su[iu])]
                ts += [Tu[iu], Tu[iu]]
            out = net(torch.cat([v[0] for v in views]), torch.cat([v[1] for v in views]), torch.cat(ts))
            loss = F.mse_loss(out[:n], Y[idx])
            if m:
                loss = loss + LAMBDA * min(1.0, step / (RAMP * steps)) * F.mse_loss(out[n:n + m], out[n + m:])
            step += 1
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
    return net


@torch.no_grad()
def predict(net, X, S, T):
    net.eval()
    X = features(X)
    views = (X, X.flip(3), X.flip(2), X.flip(2).flip(3))
    return torch.stack([net(v, S, T) for v in views]).mean(0).cpu().numpy()


def main():
    data, work = Path(sys.argv[1]), Path(sys.argv[2])
    dev = torch.device("cuda")
    tr, te = pd.read_csv(data / "train.csv"), pd.read_csv(data / "test.csv")

    def tensor(a):
        return torch.tensor(np.asarray(a, dtype=np.float32), device=dev)

    X, Xt = (tensor(np.load(data / f"{s}_geometry.npz")["phi"]) for s in ("train", "test"))
    S, T, Y = tensor(tr["scale_patch"]), tensor(tr["t_sub_um"]), tensor(tr[COLS])
    St, Tt = tensor(te["scale_patch"]), tensor(te["t_sub_um"])
    folds = np.load(work / "folds.npy")
    for k, (fold, seed) in enumerate(JOBS):
        try:
            os.close(os.open(work / f"claim_{k}", os.O_CREAT | os.O_EXCL))
        except FileExistsError:
            continue
        t0 = time.time()
        fit_idx = torch.tensor(np.flatnonzero(folds != fold), device=dev)
        val_idx = torch.tensor(np.flatnonzero(folds == fold), device=dev)
        U, Su, Tu = (torch.cat([a[val_idx], b]) for a, b in ((X, Xt), (S, St), (T, Tt)))
        net = fit(X[fit_idx], S[fit_idx], T[fit_idx], Y[fit_idx], U, Su, Tu, seed)
        res = {"test": predict(net, Xt, St, Tt), "a": net.a.item(), "b": net.b.item()}
        if fold >= 0:
            res["oof"] = predict(net, X[val_idx], S[val_idx], T[val_idx])
        np.savez(work / f"tmp_{k}.npz", **res)
        os.replace(work / f"tmp_{k}.npz", work / f"job_{k}.npz")
        print(f"job {k} fold {fold} seed {seed}: {time.time() - t0:.0f}s a {res['a']:.3f} b {res['b']:.3f}", flush=True)


if __name__ == "__main__":
    main()
