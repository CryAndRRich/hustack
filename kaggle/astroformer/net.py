import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn

from preprocess import PERIODS, T, augment, features, inject, load

BATCH = 48
LR = 2e-3
W = 10
BASE = dict(chans=(0, 1, 2, 3), aux=0.25, ch=64, ev=8, epochs=80, aug=False, cand=False)
LADDER = [("C0", {}), ("C1", dict(aug=True)), ("C2", dict(cand=True))]
ARMS = {}
for name, change in LADDER:
    BASE = {**BASE, **change}
    ARMS[name] = BASE
SEEDS = (0, 1, 2)
JOBS = [(a, s, tv) for a in reversed(ARMS) for tv in (False, True) for s in SEEDS]


def fold_matrix():
    nph = int(np.ceil(PERIODS.max()))
    M = np.zeros((T, len(PERIODS), nph), np.float32)
    bias = np.full((len(PERIODS), nph), -1e4, np.float32)
    for i, P in enumerate(PERIODS):
        for ph in range(int(np.ceil(P))):
            idx = np.rint(ph + P * np.arange(int(np.ceil(T / P)))).astype(int)
            idx = idx[idx < T]
            M[idx, i, ph] = 1 / np.sqrt(len(idx))
            bias[i, ph] = 0
    return torch.tensor(M.reshape(T, -1)), torch.tensor(bias.reshape(-1))


class Block(nn.Module):
    def __init__(self, ch, dilation):
        super().__init__()
        self.net = nn.Sequential(nn.BatchNorm1d(ch), nn.GELU(), nn.Conv1d(ch, ch, 5, padding=2 * dilation, dilation=dilation),
                                 nn.BatchNorm1d(ch), nn.GELU(), nn.Conv1d(ch, ch, 5, padding=2 * dilation, dilation=dilation))

    def forward(self, x):
        return x + self.net(x)


def window_encoder(c_in):
    return nn.Sequential(nn.Conv1d(c_in, 32, 5, padding=2), nn.GELU(), nn.Conv1d(32, 32, 5, padding=2), nn.GELU(),
                         nn.Flatten(), nn.Linear(32 * (2 * W + 1), 32))


class FoldNet(nn.Module):
    def __init__(self, n_in, n_scal, ch, ev, cand):
        super().__init__()
        self.local = nn.Sequential(nn.Conv1d(n_in, ch, 7, padding=3), *[Block(ch, d) for d in (1, 2, 4, 8)],
                                   nn.BatchNorm1d(ch), nn.GELU(), nn.Conv1d(ch, ev, 1))
        M, bias = fold_matrix()
        self.register_buffer("M", M)
        self.register_buffer("bias", bias)
        self.register_buffer("periods", torch.tensor(PERIODS, dtype=torch.float32))
        self.cand = cand
        extra = 0
        if cand:
            self.per_transit, self.folded = window_encoder(n_in), window_encoder(n_in)
            self.pconv = nn.Sequential(nn.Conv1d(ev, 16, 9, padding=4), nn.GELU(), nn.Conv1d(16, 16, 9, padding=4))
            extra = 3 * 32 + 16 + 1
        self.head = nn.Sequential(nn.Linear(5 * ev + n_scal + extra, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x, scal):
        e = self.local(x)
        B, C = e.shape[:2]
        g = (e.reshape(-1, T) @ self.M + self.bias).reshape(B, C, len(PERIODS), -1)
        f, phase = g.max(-1)
        md = f.median(-1, keepdim=True).values
        mad = (f - md).abs().median(-1).values
        top = f.amax(-1)
        sde = ((top - md[..., 0]) / (1.4826 * mad + 1e-2)).clamp(-30, 30)
        feats = [top, sde, f.logsumexp(-1) - np.log(len(PERIODS)), e.amax(-1), e.mean(-1), scal]
        if self.cand:
            ip = f[:, 0].argmax(-1)
            P = self.periods[ip]
            ph = phase[:, 0].gather(1, ip[:, None]).float()
            K = int(np.ceil(T / PERIODS.min()))
            tk = torch.round(ph + P[:, None] * torch.arange(K, device=x.device))
            valid = (tk < T).float()
            idx = tk.long().clamp(max=T - 1)[:, :, None] + torch.arange(2 * W + 1, device=x.device)
            n_in = x.shape[1]
            loc = nn.functional.pad(x, (W, W)).gather(2, idx.reshape(B, 1, -1).expand(-1, n_in, -1))
            loc = loc.reshape(B, n_in, K, 2 * W + 1).transpose(1, 2)
            z = self.per_transit(loc.reshape(B * K, n_in, -1)).reshape(B, K, -1)
            m, cnt = valid[..., None], valid.sum(1, keepdim=True)
            mean = (z * m).sum(1) / cnt
            spread = ((z - mean[:, None]) ** 2 * m).sum(1).div(cnt).add(1e-6).sqrt()
            stacked = (loc * m[..., None]).sum(1) / cnt[..., None]
            feats += [mean, spread, self.folded(stacked), self.pconv(f).amax(-1), ((P - 52) / 108)[:, None]]
        return self.head(torch.cat(feats, 1)).squeeze(1), e[:, 0]


def predict(net, x, s):
    net.eval()
    out = []
    with torch.inference_mode():
        for xb, sb in zip(x.split(128), s.split(128)):
            out.append(((net(xb, sb)[0] + net(xb.flip(-1), sb)[0]) / 2).cpu().numpy())
    return np.concatenate(out)


def fit(cfg, seed, flux, y, mu, sd, dev):
    chans = list(cfg["chans"])
    g = torch.Generator(device=dev)
    g.manual_seed(seed)
    torch.manual_seed(seed)

    def prep(f):
        x, s = features(f)
        return x[:, chans], (s - mu) / sd

    net = FoldNet(len(chans), len(mu), cfg["ch"], cfg["ev"], cfg["cand"]).to(dev)
    per_epoch = len(y) // BATCH
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=cfg["epochs"] * per_epoch)
    bce = nn.BCEWithLogitsLoss()
    for epoch in range(cfg["epochs"]):
        net.train()
        for b in torch.rand(len(y), generator=g, device=dev).argsort()[: per_epoch * BATCH].reshape(-1, BATCH):
            fb, yb = flux[b], y[b]
            bg = fb[yb == 0]
            if cfg["aug"]:
                bg = augment(bg, g)
            inj, shape = inject(bg, g)
            zeros, ones = torch.zeros(len(bg), device=dev), torch.ones(len(bg), device=dev)
            rows = [fb, bg, inj] if cfg["aug"] else [fb, inj]
            labels = [yb, zeros, ones] if cfg["aug"] else [yb, ones]
            masks = [torch.zeros(BATCH, T, device=dev)] + ([torch.zeros_like(shape, dtype=torch.float32)] if cfg["aug"] else []) + [(shape > 0.5).float()]
            known = [yb == 0] + ([ones.bool()] if cfg["aug"] else []) + [ones.bool()]
            x, s = prep(torch.cat(rows))
            yt, mask, known = torch.cat(labels).float(), torch.cat(masks), torch.cat(known)
            flip = torch.rand(len(yt), generator=g, device=dev) < 0.5
            x[flip], mask[flip] = x[flip].flip(-1), mask[flip].flip(-1)
            logit, ev = net(x, s)
            loss = bce(logit, yt) + cfg["aux"] * bce(ev[known], mask[known])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        if (epoch + 1) % 20 == 0:
            print(f"  epoch {epoch + 1}: loss {loss.item():.3f}", flush=True)
    return net, prep


def main():
    data, work = Path(sys.argv[1]), Path(sys.argv[2])
    dev = torch.device("cuda")
    torch.set_num_threads(1)
    tensor = lambda a: torch.tensor(a, device=dev)
    tr_f, tr_y, _ = load(data, "train")
    va_f, va_y, _ = load(data, "validation")
    te_f, _, _ = load(data, "test")
    tr_f, va_f, te_f, tr_y_t, va_y_t = tensor(tr_f), tensor(va_f), tensor(te_f), tensor(tr_y), tensor(va_y)
    neg = va_f[va_y_t == 0]
    g = torch.Generator(device=dev)
    g.manual_seed(999)
    synth = torch.cat([neg, inject(neg, g)[0]])
    synth_y = np.r_[np.zeros(len(neg)), np.ones(len(neg))]
    for k, (arm, seed, tv) in enumerate(JOBS):
        try:
            os.close(os.open(work / f"claim_{k}", os.O_CREAT | os.O_EXCL))
        except FileExistsError:
            continue
        t0 = time.time()
        print(f"job {k} arm {arm} seed {seed} train+val {tv}", flush=True)
        flux, y = (torch.cat([tr_f, va_f]), torch.cat([tr_y_t, va_y_t])) if tv else (tr_f, tr_y_t)
        s_fit = torch.cat([features(c)[1] for c in flux.split(256)])
        net, prep = fit(ARMS[arm], seed, flux, y, s_fit.mean(0), s_fit.std(0), dev)
        res = {"val": predict(net, *prep(va_f)), "test": predict(net, *prep(te_f)), "synth": predict(net, *prep(synth))}
        np.savez(work / f"tmp_{k}.npz", **res)
        os.replace(work / f"tmp_{k}.npz", work / f"job_{k}.npz")
        print(f"job {k} arm {arm} seed {seed} tv {tv}: val AP {average_precision_score(va_y, res['val']):.4f}"
              f" synth AP {average_precision_score(synth_y, res['synth']):.4f} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
