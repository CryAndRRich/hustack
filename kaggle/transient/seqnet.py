import numpy as np

from features import ffill

CHANNELS = ["V4", "V7", "V9", "theta4", "theta9", "f4", "f7", "f9", "P_L4-5", "P_L4-6", "P_L7-8"]


def sequences(X, ch):
    get = lambda c: X[:, :, ch.index(c)]
    out = []
    for c in CHANNELS:
        raw = get(c)
        if c in ("theta4", "theta9"):
            raw = raw - get("theta7")
        v = ffill(raw)
        pre = np.nanmedian(v[:, :20], 1, keepdims=True)
        v = v / pre - 1 if c.startswith("V") else (v - 50.0 if c.startswith("f") else v - pre)
        out.append(v)
        out.append(np.isfinite(raw).astype(np.float64))
    return np.nan_to_num(np.stack(out, 2)).astype(np.float32)


def fit_predict_factory(seq, scale):
    def fit_predict(cfg, tr, cols, Xs):
        import torch
        import torch.nn as nn
        torch.manual_seed(cfg.seed)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        S = {k: torch.tensor(v / scale, device=dev) for k, v in seq.items()}
        mu = tr[cols].mean().values
        sd = tr[cols].std().replace(0, 1).fillna(1).values

        def tensors(df):
            tab = np.nan_to_num((df[cols].values - mu) / sd).clip(-10, 10).astype(np.float32)
            k = np.round(df["cut_s"].values * 100).astype(int) + 20
            return (torch.tensor(tab, device=dev), torch.tensor(df["src"].values, device=dev),
                    torch.tensor(df["event"].values, device=dev), torch.tensor(k, device=dev))

        class Net(nn.Module):
            def __init__(self, n_ch, n_tab):
                super().__init__()
                self.gru = nn.GRU(n_ch, cfg.gru_hidden, batch_first=True)
                self.tab = nn.Sequential(nn.Linear(n_tab, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 64), nn.ReLU())
                self.head = nn.Sequential(nn.Linear(cfg.gru_hidden + 64, 64), nn.ReLU(), nn.Linear(64, 1))

            def forward(self, tab, src, event, k):
                x = torch.where(src[:, None, None] == 0, S["train"][event.clamp(max=len(S["train"]) - 1)],
                                S["test"][event.clamp(max=len(S["test"]) - 1)])
                h, _ = self.gru(x[:, :int(k.max()) + 1])
                h = h[torch.arange(len(k), device=dev), k]
                return self.head(torch.cat([h, self.tab(tab)], 1))[:, 0]

        tab, src, event, k = tensors(tr)
        y = torch.tensor(tr["y"].values, dtype=torch.float32, device=dev)
        w = torch.tensor(tr["w"].values, dtype=torch.float32, device=dev)
        net = Net(next(iter(S.values())).shape[2], len(cols)).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=cfg.gru_lr, weight_decay=1e-4)
        steps = cfg.gru_epochs * int(np.ceil(len(y) / cfg.gru_batch))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.gru_lr, total_steps=steps)
        loss_fn = nn.BCEWithLogitsLoss(reduction="none")
        net.train()
        for _ in range(cfg.gru_epochs):
            perm = torch.randperm(len(y), device=dev)
            for b in range(0, len(y), cfg.gru_batch):
                i = perm[b:b + cfg.gru_batch]
                loss = (loss_fn(net(tab[i], src[i], event[i], k[i]), y[i]) * w[i]).sum() / w[i].sum()
                opt.zero_grad()
                loss.backward()
                opt.step()
                sched.step()
        net.eval()
        outs = []
        with torch.no_grad():
            for X in Xs:
                t2, s2, e2, k2 = tensors(X)
                p = [torch.sigmoid(net(t2[b:b + 4096], s2[b:b + 4096], e2[b:b + 4096], k2[b:b + 4096]))
                     for b in range(0, len(t2), 4096)]
                outs.append(torch.cat(p).cpu().numpy())
        return outs
    return fit_predict
