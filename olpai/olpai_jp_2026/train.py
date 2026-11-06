import copy
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

EPOCHS = int(os.environ.get("EPOCHS", 40))
BATCH = 128
HIDDEN = 128
NOISE = 0.1
N_META = 6
AUX_W = (1.0, 0.5, 0.5)
DEVICE = torch.device("cuda")
ARCHS = {"gru": 2e-3, "tcn": 2e-3, "tf": 1e-3}


class TBlock(nn.Module):
    def __init__(self, h, d):
        super().__init__()
        self.c1 = nn.Conv1d(h, h, 5, padding=2 * d, dilation=d)
        self.c2 = nn.Conv1d(h, h, 5, padding=2 * d, dilation=d)
        self.norm = nn.GroupNorm(1, h)
        self.drop = nn.Dropout(0.1)

    def forward(self, x, mask):
        y = self.c2(self.drop(F.gelu(self.c1(x * mask))))
        return self.norm(x + y) * mask


class Net(nn.Module):
    def __init__(self, arch, n_in, n_mice, n_md):
        super().__init__()
        self.arch = arch
        self.emb = nn.Embedding(n_mice, 8)
        self.md = nn.Embedding(n_md, 8)
        self.inp = nn.Sequential(nn.Linear(n_in + 16, HIDDEN), nn.GELU(), nn.Dropout(0.1))
        if arch == "gru":
            self.body = nn.GRU(HIDDEN, HIDDEN, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
            out = 2 * HIDDEN
        elif arch == "tcn":
            self.body = nn.ModuleList([TBlock(HIDDEN, d) for d in (1, 2, 4, 8, 1, 2)])
            out = HIDDEN
        else:
            self.pos = nn.Parameter(torch.zeros(1, 40, HIDDEN))
            layer = nn.TransformerEncoderLayer(HIDDEN, 4, 2 * HIDDEN, 0.1, "gelu", batch_first=True, norm_first=True)
            self.body = nn.TransformerEncoder(layer, 3, enable_nested_tensor=False)
            out = HIDDEN
        self.head = nn.Sequential(nn.Linear(out, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, 3))

    def forward(self, x, n_cpu, m, md):
        T = x.shape[1]
        valid = torch.arange(T, device=x.device)[None] < n_cpu.to(x.device)[:, None]
        e = torch.cat([self.emb(m), F.dropout(self.md(md), 0.2, self.training)], -1)
        h = self.inp(torch.cat([x, e[:, None].expand(-1, T, -1)], -1))
        if self.arch == "gru":
            h = pack_padded_sequence(h, n_cpu, batch_first=True, enforce_sorted=False)
            h = pad_packed_sequence(self.body(h)[0], batch_first=True, total_length=T)[0]
        elif self.arch == "tcn":
            mask = valid[:, None].float()
            h = h.transpose(1, 2)
            for blk in self.body:
                h = blk(h, mask)
            h = h.transpose(1, 2)
        else:
            h = self.body(h + self.pos[:, :T], src_key_padding_mask=~valid)
        return self.head(h)


def predict(net, d, idx):
    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(idx), 1024):
            b = idx[i:i + 1024]
            out.append(net(d["X"][b], d["n"][b], d["M"][b], d["MD"][b])[..., 0].cpu().numpy())
    return np.concatenate(out)


def masked_mse(p, y, n):
    mask = np.arange(p.shape[1])[None] < n[:, None]
    return float(((p - y) ** 2)[mask].mean())


def targets(Y, n, sd=None):
    T = Y.shape[1]
    valid = (torch.arange(T)[None] < n[:, None]).to(DEVICE)
    d1 = torch.zeros_like(Y)
    d1[:, 1:] = Y[:, 1:] - Y[:, :-1]
    d2 = torch.zeros_like(Y)
    d2[:, 2:] = d1[:, 2:] - d1[:, 1:-1]
    v1, v2 = valid.clone(), valid.clone()
    v1[:, :1] = False
    v2[:, :2] = False
    tg = [Y, d1, d2]
    sd = sd or [float(t[v].std()) for t, v in zip(tg, (valid, v1, v2))]
    return torch.stack([t / s for t, s in zip(tg, sd)], -1), torch.stack([valid, v1, v2], -1), sd


def fit(d, arch, a, b, seed, tag, deadline):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = Net(arch, d["X"].shape[2], d["n_mice"], d["n_md"]).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), ARCHS[arch], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, ARCHS[arch], total_steps=EPOCHS * -(-len(a) // BATCH), pct_start=0.1)
    w = torch.tensor(AUX_W, device=DEVICE)
    noise_cols = torch.ones(d["X"].shape[2], device=DEVICE)
    noise_cols[-N_META:] = 0
    best, state = np.inf, None
    for ep in range(EPOCHS):
        t = time.time()
        net.train()
        perm = a[rng.permutation(len(a))]
        for i in range(0, len(perm), BATCH):
            j = perm[i:i + BATCH]
            x = d["X"][j]
            x = x + NOISE * torch.randn_like(x) * noise_cols
            err = (net(x, d["n"][j], d["M"][j], d["MD"][j]) - d["TG"][j]) ** 2
            v = d["V"][j]
            loss = sum(w[c] * err[..., c][v[..., c]].mean() for c in range(3))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
        score = masked_mse(predict(net, d, b) * d["sd"][0], d["y_np"][b], d["n_np"][b])
        if score < best:
            best, state = score, copy.deepcopy(net.state_dict())
        if ep < 2 or ep % 10 == 0 or ep == EPOCHS - 1:
            print(f"{tag} ep {ep}: valid mse {score:.4f}, best {best:.4f}, {time.time() - t:.2f}s/epoch", flush=True)
        if time.time() > deadline:
            print(f"{tag} deadline hit at epoch {ep}, keeping best {best:.4f}", flush=True)
            break
    net.load_state_dict(state)
    return net, best


def run():
    torch.set_num_threads(1)
    assert torch.cuda.is_available(), "GPU worker started without a GPU"
    z = np.load(os.environ["DATA"])
    parts, deadline = Path(os.environ["PARTS"]), float(os.environ["DEADLINE"])
    ntr, nte = torch.from_numpy(z["ntr"]), torch.from_numpy(z["nte"])
    Ytr = torch.from_numpy(z["ytr"]).to(DEVICE)
    TG, V, sd = targets(Ytr, ntr)
    g = lambda k: torch.from_numpy(z[k]).to(DEVICE)
    tr = dict(X=g("Xtr"), n=ntr, M=g("mtr"), MD=g("mdtr"), TG=TG, V=V)
    te = dict(X=g("Xte"), n=nte, M=g("mte"), MD=g("mdte"))
    both = {k: torch.cat([tr[k], te[k]]) for k in te}
    common = dict(sd=sd, y_np=z["ytr"], n_np=z["ntr"], n_mice=int(z["n_mice"]), n_md=int(z["n_md"]))
    fold, W = z["fold"], len(z["ntr"])
    for job in os.environ["JOBS"].split(","):
        arm, k, s = job.split(":")
        k, s = int(k), int(s)
        student, arch = arm.startswith("s_"), arm.removeprefix("s_")
        if time.time() > deadline - 30:
            print(f"[{arm} fold {k} seed {s}] skipped: past deadline", flush=True)
            continue
        a, b = np.where(fold != k)[0], np.where(fold == k)[0]
        if student:
            teach = np.load(parts / f"teach_{k}.npz")
            Y = Ytr.clone()
            Y[b] = torch.from_numpy(teach["oof"][b]).to(DEVICE)
            TGs, Vs, _ = targets(torch.cat([Y, torch.from_numpy(teach["test"]).to(DEVICE)]), both["n"], sd)
            d = common | both | dict(TG=TGs, V=Vs)
            train_idx = np.concatenate([a, b, W + np.arange(len(z["nte"]))])
        else:
            d, train_idx = common | tr, a
        t = time.time()
        seed = 100 * s + k + (1000 if student else 0)
        net, best = fit(d, arch, train_idx, b, seed, f"[{arm} fold {k} seed {s}]", deadline)
        oof = np.full(z["ytr"].shape, np.nan, np.float32)
        oof[b] = predict(net, d, b) * sd[0]
        test = predict(net, common | te, np.arange(len(z["nte"]))) * sd[0]
        np.save(parts / f"oof_{arm}_{k}_{s}.npy", oof)
        np.save(parts / f"test_{arm}_{k}_{s}.npy", test.astype(np.float32))
        print(f"[{arm} fold {k} seed {s}] done: valid mse {best:.4f} [{time.time() - t:.0f}s]", flush=True)


if __name__ == "__main__":
    run()
