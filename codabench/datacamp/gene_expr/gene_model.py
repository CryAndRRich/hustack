import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from common.runtime import log

RANK = 32
HIDDEN = (512, 256)
BATCH = 500_000
STEPS1, STEPS2, EVAL_EVERY, PATIENCE = 3000, 2000, 200, 4
LR_ENC, LR_RUN, WD, L2_V, L2_U, CLIP = 1e-3, 1e-2, 1e-4, 1e-4, 1e-4, 1.0
VAL_CELLS = 1_000_000
ETA_MAX, LOG_R_MIN, LOG_R_MAX, NLL_CAP = 30.0, -6.0, 7.0, 100.0


def bound(x, lo, hi):
    return x + (x.clamp(lo, hi) - x).detach()


def nb_nll(y, eta, log_r, cap=True):
    y, eta, log_r = y.double(), bound(eta.double(), -ETA_MAX, ETA_MAX), bound(log_r.double(), LOG_R_MIN, LOG_R_MAX)
    r = log_r.exp()
    lse = torch.logaddexp(log_r, eta)
    nll = (torch.lgamma(y + 1) + torch.lgamma(r) - torch.lgamma(y + r) - r * (log_r - lse) - y * (eta - lse))
    return nll.clamp(max=NLL_CAP) + (0.0 if cap else torch.log1p((nll - NLL_CAP).clamp(min=0.0)))


class Encoder(nn.Module):

    def __init__(self, d_in, n_genome):
        super().__init__()
        self.inp = nn.Linear(d_in, HIDDEN[0])
        self.genome = nn.Embedding(n_genome, HIDDEN[0])
        nn.init.zeros_(self.genome.weight)
        layers, d = [nn.GELU(), nn.Dropout(0.1)], HIDDEN[0]
        for h in HIDDEN[1:]:
            layers += [nn.Linear(d, h), nn.GELU(), nn.Dropout(0.1)]
            d = h
        layers.append(nn.Linear(d, 2 + RANK))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, genome):
        return self.net(self.inp(x) + self.genome(genome))


class Runs(nn.Module):
    def __init__(self, R):
        super().__init__()
        self.base = nn.Parameter(torch.zeros(()))
        self.b = nn.Parameter(torch.zeros(R))
        self.d = nn.Parameter(torch.zeros(R))
        self.v = nn.Parameter(torch.randn(R, RANK) * 0.01)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--variant", required=True, choices=("esm", "prom"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    steps1, steps2, every = (10, 10, 5) if args.smoke else (STEPS1, STEPS2, EVAL_EVERY)
    data, dev = Path(args.data), torch.device("cuda")
    torch.manual_seed(0)
    gen = torch.Generator(device=dev).manual_seed(0)

    X = np.load(data / "X_esm.npy", mmap_mode="r")
    if args.variant == "prom":
        X = np.hstack([X, np.load(data / "X_prom.npy", mmap_mode="r")])
    X = torch.as_tensor(np.ascontiguousarray(X), dtype=torch.float16, device=dev)
    genome = torch.as_tensor(np.load(data / "gene_genome.npy"), dtype=torch.long, device=dev)
    fold_np = np.load(data / "gene_fold.npy")
    fold = torch.as_tensor(fold_np, dtype=torch.long, device=dev)
    K = int(fold_np.max()) + 1
    cells = {k: torch.as_tensor(np.load(data / f"train_{k}.npy"), device=dev) for k in ("g", "r", "y", "off")}
    evc = {k: torch.as_tensor(np.load(data / f"evalc_{k}.npy"), device=dev) for k in ("g", "r", "y", "off")}
    gi, ri, y, off = (cells[k] for k in ("g", "r", "y", "off"))
    gi, ri = gi.long(), ri.long()
    N, G, R = len(y), X.shape[0], int(np.load(data / "n_runs.npy"))
    n_genome = int(genome.max()) + 1
    log(f"variant {args.variant}: genes {G:,} x {X.shape[1]} features, Train cells {N:,}, runs {R:,}, folds {K}, "
        f"eval-C cells {len(evc['y']):,}")

    def predict(enc, g):
        out = []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            for i in range(0, len(g), 200_000):
                gg = g[i:i + 200_000]
                out.append(enc(X[gg], genome[gg]).float())
        return torch.cat(out)

    def cell_nll(enc, runs, g, r, yy, oo, a_c_u=None):
        tot = 0.0
        for i in range(0, len(yy), 1_000_000):
            sl = slice(i, i + 1_000_000)
            if a_c_u is None:
                ug, inv = torch.unique(g[sl], return_inverse=True)
                o = predict(enc, ug)[inv]
            else:
                o = a_c_u[g[sl]]
            eta = oo[sl] + runs.base + o[:, 0] + runs.b[r[sl]] + (o[:, 2:] * runs.v[r[sl]]).sum(1)
            tot += float(nb_nll(yy[sl], eta, runs.d[r[sl]] + o[:, 1]).sum())
        return tot / max(len(yy), 1)

    def train(enc, runs, steps, keep=None, val=None, tag=""):
        params = [{"params": enc.parameters(), "lr": LR_ENC, "weight_decay": WD}]
        if any(p.requires_grad for p in runs.parameters()):
            params.append({"params": runs.parameters(), "lr": LR_RUN, "weight_decay": 0.0})
        opt = torch.optim.AdamW(params)
        scaler = torch.amp.GradScaler("cuda")
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[p["lr"] for p in params], total_steps=steps,
                                                    pct_start=0.05)
        best, best_state, bad, t = np.inf, None, 0, time.time()
        for step in range(1, steps + 1):
            enc.train()
            idx = torch.randint(0, N, (BATCH,), device=dev, generator=gen)
            if keep is not None:
                idx = idx[keep[gi[idx]]]
            g, r = gi[idx], ri[idx]
            ug, inv = torch.unique(g, return_inverse=True)
            with torch.autocast("cuda", dtype=torch.float16):
                o = enc(X[ug], genome[ug]).float()[inv]
            eta = off[idx] + runs.base + o[:, 0] + runs.b[r] + (o[:, 2:] * runs.v[r]).sum(1)
            nll = nb_nll(y[idx], eta, runs.d[r] + o[:, 1], cap=False)
            loss = nll.mean() + L2_U * (o[:, 2:] ** 2).sum(1).mean()
            if runs.v.requires_grad:
                loss = loss + L2_V * (runs.v[r] ** 2).sum(1).mean()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss.float()).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([q for g_ in opt.param_groups for q in g_["params"]], CLIP)
            scaler.step(opt)
            scaler.update()
            loss = nll.clamp(max=NLL_CAP).mean()
            sched.step()
            if val is not None and (step % every == 0 or step == steps):
                enc.eval()
                v = cell_nll(enc, runs, *val)
                if v < best - 1e-4:
                    best, bad = v, 0
                    best_state = {k: p.detach().clone() for k, p in enc.state_dict().items()}
                else:
                    bad += 1
                log(f"{tag} step {step}: train NLL {loss.item():.4f}, held-out NLL {v:.4f} "
                    f"({(time.time() - t) / step * 1000:.0f} ms/step)")
                if bad >= PATIENCE:
                    break
            elif step % 500 == 0 or step == steps:
                log(f"{tag} step {step}: train NLL {loss.item():.4f} ({(time.time() - t) / step * 1000:.0f} ms/step)")
        if best_state is not None:
            enc.load_state_dict(best_state)
        return best

    runs = Runs(R).to(dev)
    s = torch.randperm(N, device=dev, generator=gen)[:VAL_CELLS]
    runs.base.data.fill_(float((torch.log(y[s].float() + 0.5) - off[s].float()).mean()))
    log(f"base level {runs.base.item():.3f}")
    train(Encoder(X.shape[1], n_genome).to(dev), runs, steps1, tag="stage 1")
    for p in runs.parameters():
        p.requires_grad_(False)

    out = torch.zeros(G, 2 + RANK, device=dev)
    cold = torch.nonzero(fold < 0).squeeze(1)
    oof = []
    for k in range(K):
        in_k = fold == k
        sel = torch.nonzero(in_k[gi]).squeeze(1)
        sel = sel[torch.randperm(len(sel), device=dev, generator=gen)[:VAL_CELLS]]
        enc = Encoder(X.shape[1], n_genome).to(dev)
        best = train(enc, runs, steps2, keep=~in_k, val=(gi[sel], ri[sel], y[sel], off[sel]), tag=f"fold {k}")
        enc.eval()
        members = torch.nonzero(in_k).squeeze(1)
        out[members] = predict(enc, members)
        out[cold] += predict(enc, cold) / K
        oof.append(best)
        log(f"fold {k}: genes {len(members):,}, best held-out NLL {best:.4f}")

    log(f"out-of-fold NLL per fold {np.round(oof, 4).tolist()}, mean {np.mean(oof):.4f}")
    if len(evc["y"]):
        e = (evc["g"].long(), evc["r"].long(), evc["y"], evc["off"])
        full = cell_nll(None, runs, *e, a_c_u=out)
        level = out.clone()
        level[:, 2:] = 0
        flat = torch.zeros_like(out)
        flat[:, 0] = out[cold, 0].mean() if len(cold) else 0.0
        log(f"eval C ({len(e[2]):,} cells, never trained on): NLL full {full:.4f}, level only "
            f"{cell_nll(None, runs, *e, a_c_u=level):.4f}, constant gene {cell_nll(None, runs, *e, a_c_u=flat):.4f}")
    q = lambda v: np.round(np.percentile(v.cpu().numpy(), [1, 50, 99]), 2).tolist()
    log(f"outputs (percentiles 1/50/99): level a {q(out[:, 0])}, dispersion c {q(out[:, 1])}, |u| "
        f"{q(out[:, 2:].norm(dim=1))}; run d {q(runs.d)}, |v| {q(runs.v.norm(dim=1))}")
    out[:, 0] += runs.base
    res = out.cpu().numpy()
    assert np.isfinite(res).all(), "non-finite gene outputs"
    np.save(args.out, res.astype(np.float16))
    log(f"wrote {args.out} {res.shape}")


if __name__ == "__main__":
    main()
