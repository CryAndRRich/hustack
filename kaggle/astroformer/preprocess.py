from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

T = 512
WINDOWS = (13, 25, 49)
BINS = (2, 4, 8)
PERIODS = np.arange(52, 160.5, 0.5)


def find_data():
    kaggle = Path("/kaggle/input")
    if kaggle.exists():
        return next(kaggle.rglob("train.npz")).parent
    try:
        return Path(__file__).resolve().parent.parent / "data"
    except NameError:
        return Path.cwd() / "data"


def load(data, name):
    with np.load(data / f"{name}.npz", allow_pickle=False) as f:
        return f["flux"].astype(np.float64), (f["labels"] if "labels" in f else None), f["id"]


def med(a, dim=1):
    return a.median(dim, keepdim=True).values


def running_median(f, w):
    return F.pad(f[:, None], (w // 2, w // 2), mode="replicate")[:, 0].unfold(1, w, 1).median(-1).values


def noise_sigma(f):
    d = f.diff(dim=1)
    return (1.4826 * med((d - med(d)).abs()) / 2 ** 0.5).clamp_min(1e-7)


def features(f):
    s = noise_sigma(f)
    res = [((f - running_median(f, w)) / s).clamp(-10, 10) for w in WINDOWS]
    x = torch.stack(res + [((f - 1) / s).clamp(-10, 10)], 1)
    r = res[1]
    rms = [(b * r.reshape(len(r), -1, b).mean(2).pow(2).mean(1)).sqrt() for b in BINS]
    scal = torch.stack([s[:, 0].log(), r.pow(2).mean(1).sqrt().log()] + [v.log() for v in rms], 1)
    return x.float(), scal.float()


def uniform(g, n, lo, hi, dev):
    return lo + (hi - lo) * torch.rand(n, 1, generator=g, device=dev, dtype=torch.float64)


def inject(f, g, limb=True, snr=(4.0, 10.0), sub=8):
    n, dev = len(f), f.device
    u = lambda lo, hi: uniform(g, n, lo, hi, dev)
    P, dur, k = u(53, 152), u(1.8, 5.5), u(0.02, 0.12)
    tc = P * u(0, 1)
    b, u1, u2 = (u(0, 0.9), u(0.3, 0.6), u(0.05, 0.3)) if limb else (0.0, 0.0, 0.0)
    t = (torch.arange(T * sub, device=dev, dtype=torch.float64) + 0.5) / sub
    dt = ((t - tc + P / 2) % P - P / 2).abs()
    z = ((dt / (dur / 2)) ** 2 * ((1 + k) ** 2 - b ** 2) + b ** 2).sqrt()
    mu = (1 - z.clamp(max=1) ** 2).clamp(min=0).sqrt()
    shape = ((1 + k - z) / (2 * k)).clamp(0, 1) * (1 - u1 * (1 - mu) - u2 * (1 - mu) ** 2)
    shape = shape.reshape(n, T, sub).mean(2)
    shape = shape / shape.amax(1, keepdim=True)
    target = u(np.log(snr[0]), np.log(snr[1])).exp()
    depth = target * noise_sigma(f) / shape.pow(2).sum(1, keepdim=True).sqrt()
    out = f - depth * shape
    return out / out.mean(1, keepdim=True), shape


def augment(f, g):
    n, dev = len(f), f.device
    u = lambda lo, hi: uniform(g, n, lo, hi, dev)
    trend = running_median(f, 25)
    f = 1 + (trend - 1) * u(np.log(0.5), np.log(2.0)).exp() + (f - trend) * u(np.log(0.8), np.log(1.25)).exp()
    mode = (u(0, 3)).floor().long()[:, 0]
    pos = torch.arange(T, device=dev)
    roll = (pos + (u(0, T)).long()) % T
    f = torch.where((mode == 1)[:, None], f.gather(1, roll), f)
    other = f[torch.rand(n, generator=g, device=dev).argsort()]
    other = 1 + (other - 1) * noise_sigma(f) / noise_sigma(other)
    cut = u(128, 384).long()
    near = lambda a, lo: a.gather(1, (lo + torch.arange(8, device=dev)).clamp(0, T - 1)).mean(1, keepdim=True)
    other = other + near(f, cut - 8) - near(other, cut)
    f = torch.where((mode == 2)[:, None] & (pos >= cut), other, f)
    return f / f.mean(1, keepdim=True)
