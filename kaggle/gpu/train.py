import copy
import os
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupKFold
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

from preprocess import load

N_FOLDS = 5
EPOCHS = 40
BATCH = 16
HIDDEN = 96
LR = 2e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    def __init__(self, n_in):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(n_in, HIDDEN), nn.GELU())
        self.rnn = nn.GRU(HIDDEN, HIDDEN, num_layers=2, batch_first=True, bidirectional=True, dropout=0.1)
        self.head = nn.Linear(2 * HIDDEN, 1)

    def forward(self, x, lengths):
        h = pack_padded_sequence(self.inp(x), lengths, batch_first=True, enforce_sorted=False)
        h = pad_packed_sequence(self.rnn(h)[0], batch_first=True, total_length=x.shape[1])[0]
        return self.head(h).squeeze(-1)


def split(df, a):
    cut = np.cumsum(df.groupby("device_id", sort=True).size().to_numpy())[:-1]
    return [torch.from_numpy(x).to(DEVICE) for x in np.split(a.astype(np.float32), cut)]


def batches(seqs, idx, ys=None):
    for i in range(0, len(idx), BATCH):
        b = idx[i:i + BATCH]
        lengths = torch.tensor([len(seqs[j]) for j in b])
        x = pad_sequence([seqs[j] for j in b], batch_first=True)
        y = None if ys is None else pad_sequence([ys[j] for j in b], batch_first=True)
        yield x, y, lengths, (torch.arange(x.shape[1])[None] < lengths[:, None]).to(DEVICE)


def predict(net, seqs, idx):
    net.eval()
    out = []
    with torch.no_grad():
        for x, _, lengths, _ in batches(seqs, idx):
            p = net(x, lengths).cpu()
            out += [p[k, :n].numpy() for k, n in enumerate(lengths.tolist())]
    return out


def fit(seqs, ys, a, b, seed, tag):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = Net(seqs[0].shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=EPOCHS * -(-len(a) // BATCH), pct_start=0.1)
    yb = torch.cat([ys[j] for j in b]).cpu().numpy()
    best, state = np.inf, None
    for ep in range(EPOCHS):
        t = time.time()
        net.train()
        for x, y, lengths, mask in batches(seqs, rng.permutation(a), ys):
            loss = ((net(x, lengths) - y) ** 2)[mask].mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
        score = float(np.sqrt(np.mean((np.concatenate(predict(net, seqs, b)) - yb) ** 2)))
        if score < best:
            best, state = score, copy.deepcopy(net.state_dict())
        if ep < 2 or ep % 10 == 0 or ep == EPOCHS - 1:
            print(f"{tag} ep {ep}: valid {score:.4f} (std units), best {best:.4f}, {time.time() - t:.1f}s/epoch",
                  flush=True)
    net.load_state_dict(state)
    return net


def run():
    torch.set_num_threads(1)
    tr, Xtr, ytr, te, Xte = load()
    mu, sd = Xtr.mean(), Xtr.std() + 1e-6
    ymu, ysd = ytr.mean(), ytr.std()
    seqs = split(tr, ((Xtr - mu) / sd).to_numpy())
    ys = split(tr, (ytr - ymu) / ysd)
    test_seqs = split(te, ((Xte - mu) / sd).to_numpy())
    devices = np.sort(tr["device_id"].unique())
    ends = np.cumsum([len(s) for s in seqs])
    folds = [[np.searchsorted(devices, tr["device_id"].iloc[r].unique()) for r in f]
             for f in GroupKFold(N_FOLDS).split(Xtr, ytr, tr["device_id"])]
    parts = Path(os.environ["PARTS"])
    for job in os.environ["JOBS"].split(","):
        k, s = map(int, job.split(":"))
        a, b = folds[k]
        t = time.time()
        net = fit(seqs, ys, a, b, 100 * s + k, f"[fold {k} seed {s}]")
        oof = np.full(len(ytr), np.nan, np.float32)
        for j, p in zip(b, predict(net, seqs, b)):
            oof[ends[j] - len(p):ends[j]] = p * ysd + ymu
        test = np.concatenate(predict(net, test_seqs, np.arange(len(test_seqs)))) * ysd + ymu
        np.save(parts / f"oof_{k}_{s}.npy", oof)
        np.save(parts / f"test_{k}_{s}.npy", test.astype(np.float32))
        m = ~np.isnan(oof)
        print(f"[fold {k} seed {s}] done: rmse {np.sqrt(np.mean((oof[m] - ytr[m]) ** 2)):.4f} "
              f"[{time.time() - t:.0f}s]", flush=True)


if __name__ == "__main__":
    run()
