import math
import time

import numpy as np
import torch

from benchmark_utils.base_solver import CRMSolver

ARM = {'interp': 'cp', 'extrap': 'nf_blend'}
ARMS = {"v2.0": {"interp": "rho", "extrap": "linear"},
        "v2.1": {"interp": "cp", "extrap": "linear"},
        "v2.2": {"interp": "rho", "extrap": "ls3"},
        "v2.3": {"interp": "rho", "extrap": "shift"},
        "v2.4": {"interp": "rho", "extrap": "nf_blend"},
        "v2.5": {"interp": "rho", "extrap": "shift_pure"}}
GAMMA = 1.4
COMPONENT_WEIGHTS = {"wing": 0.3, "pylon": 0.3, "fuselage": 0.2, "nacelle": 0.2}
SHIFT_GRID = np.round(np.arange(-4.0, 4.0001, 0.05), 2)

NF_SECONDS = 1500
NF_FREQS, NF_SIGMA = 128, 1.0
NF_WIDTH, NF_DEPTH = 512, 5
NF_RUNS, NF_POINTS = 4, 16384
NF_LR, NF_WARMUP = 1e-3, 0.02
NF_CHUNK = 65536
SEED = 0


def log(*a):
    print("[onera_crm]", *a, flush=True)


def point_weights(components, component_names):
    w = np.zeros(len(components))
    for cid, name in enumerate(component_names):
        mask = components == cid
        if name in COMPONENT_WEIGHTS and mask.any():
            w[mask] = COMPONENT_WEIGHTS[name] / mask.sum()
    return w / w.sum()


def to_space(rho, mach, space):
    if space == "rho":
        return rho
    return 2.0 / (GAMMA * mach ** 2) * (np.power(rho, GAMMA) - 1.0)


def from_space(q, mach, space):
    if space == "rho":
        return q
    return np.power(np.maximum(1.0 + 0.5 * GAMMA * mach ** 2 * q, 1e-6), 1.0 / GAMMA)


class LineModel:

    def __init__(self, meta, interp, extrap):
        self.meta, self.interp, self.extrap = meta, interp, extrap
        self.w = point_weights(np.asarray(meta["components"]), list(meta["component_names"]))
        self.nf = None

    def fit(self, conditions, fields):
        conditions = np.asarray(conditions, np.float64)
        self.lines = {}
        for m in np.unique(conditions[:, 0]):
            for p in np.unique(conditions[conditions[:, 0] == m, 2]):
                sel = np.flatnonzero((conditions[:, 0] == m) & (conditions[:, 2] == p))
                sel = sel[np.argsort(conditions[sel, 1])]
                self.lines[(m, p)] = (conditions[sel, 1], np.asarray(fields[sel], np.float32))
        self.machs = np.unique(conditions[:, 0])
        log(f"LineModel interp={self.interp} extrap={self.extrap}: {len(self.lines)} lines, Mach {self.machs.tolist()}")
        if self.extrap in ("shift", "shift_pure"):
            self.fit_shift()
        if self.extrap == "nf_blend":
            self.train_data = (conditions, fields)
        return self

    def line_pi(self, mach, pi):
        pis = np.array([p for (m, p) in self.lines if m == mach])
        return pis[np.argmin(np.abs(np.log2(pis) - np.log2(pi)))]

    def at_aoa(self, mach, pi, aoa, space="rho"):
        a, f = self.lines[(mach, self.line_pi(mach, pi))]
        aoa = float(np.clip(aoa, a[0], a[-1]))
        j = int(np.clip(np.searchsorted(a, aoa), 1, len(a) - 1))
        t = (aoa - a[j - 1]) / (a[j] - a[j - 1])
        lo, hi = (to_space(np.asarray(f[k], np.float64), mach, space) for k in (j - 1, j))
        return (1 - t) * lo + t * hi

    def fit_shift(self):
        top = self.machs[-1]
        grid = self.lines[(top, self.line_pi(top, 1.0))][0]
        rates = []
        for lo, hi in ((self.machs[-3], self.machs[-2]), (self.machs[-2], self.machs[-1])):
            pis = sorted(p for (m, p) in self.lines if m == hi)
            a_hi = self.lines[(hi, pis[0])][0]
            a_lo = self.lines[(lo, pis[0])][0]
            best = []
            for a in a_hi:
                ds = SHIFT_GRID[(a + SHIFT_GRID >= a_lo[0]) & (a + SHIFT_GRID <= a_lo[-1])]
                target = [self.at_aoa(hi, p, a) for p in pis]
                err = [sum(np.abs(self.at_aoa(lo, p, a + d) - y) @ self.w for p, y in zip(pis, target)) for d in ds]
                j = int(np.argmin(err))
                best.append(ds[j] if 0 < j < len(ds) - 1 else np.nan)
            rate = np.array(best) / (hi - lo)
            log(f"shift {lo:.2f}->{hi:.2f}: AoA {a_hi.tolist()} -> k {np.round(rate, 1).tolist()}")
            ok = np.isfinite(rate)
            rates.append(np.interp(grid, a_hi[ok], rate[ok]) if ok.any() else np.zeros(len(grid)))
        self.shift_aoa, self.shift_k = grid, np.mean(rates, 0)
        log(f"shift k(a) on {grid.tolist()}: {np.round(self.shift_k, 1).tolist()}")

    def interpolate(self, m, a, p):
        lines = self.machs
        hi = int(np.clip(np.searchsorted(lines, m), 1, len(lines) - 1))
        m0, m1 = lines[hi - 1], lines[hi]
        t = (m - m0) / (m1 - m0)
        space = self.interp if m <= lines[-1] else "rho"
        q = (1 - t) * self.at_aoa(m0, p, a, space) + t * self.at_aoa(m1, p, a, space)
        return from_space(q, m, space)

    def extrapolate(self, m, a, p):
        if self.extrap in ("linear", "nf_blend"):
            return self.interpolate(m, a, p)
        if self.extrap == "ls3":
            ms = self.machs[-3:]
            v = np.stack([self.at_aoa(mj, p, a) for mj in ms])
            dm = ms - ms.mean()
            slope = np.tensordot(dm, v - v.mean(0), 1) / (dm @ dm)
            return v.mean(0) + slope * (m - ms.mean())
        prev, top = self.machs[-2], self.machs[-1]
        k = np.interp(a, self.shift_aoa, self.shift_k)
        a1 = a + k * (m - top)
        f_top = self.at_aoa(top, p, a1)
        if self.extrap == "shift_pure":
            return f_top
        return f_top + (m - top) / (top - prev) * (f_top - self.at_aoa(prev, p, a1 + k * (top - prev)))

    def predict(self, conditions):
        conditions = np.asarray(conditions, np.float64)
        n_points = len(self.w)
        out = np.empty((len(conditions), n_points), np.float32)
        above = conditions[:, 0] > self.machs[-1] + 1e-9
        for i, (m, a, p) in enumerate(conditions):
            out[i] = self.extrapolate(m, a, p) if above[i] else self.interpolate(m, a, p)
        if self.extrap == "nf_blend" and above.any():
            if self.nf is None:
                self.nf = NeuralField(self.meta).fit(*self.train_data)
            out[above] = 0.5 * out[above] + 0.5 * self.nf.predict(conditions[above])
        return out


class Field(torch.nn.Module):
    def __init__(self, n_in):
        super().__init__()
        layers, d = [], n_in
        for _ in range(NF_DEPTH):
            layers += [torch.nn.Linear(d, NF_WIDTH), torch.nn.LeakyReLU()]
            d = NF_WIDTH
        layers.append(torch.nn.Linear(d, 1))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class NeuralField:

    def __init__(self, meta):
        self.geometry = np.asarray(meta["geometry"], np.float32)
        self.w = point_weights(np.asarray(meta["components"]), list(meta["component_names"]))
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.dev.type == "cuda" else torch.float32

    def point_features(self):
        g = torch.Generator().manual_seed(SEED)
        xyz = self.geometry[:, :3]
        lo, hi = xyz.min(0), xyz.max(0)
        x = torch.from_numpy(2 * (xyz - lo) / (hi - lo) - 1).to(self.dev)
        b = (torch.randn(3, NF_FREQS, generator=g) * NF_SIGMA).to(self.dev)
        proj = 2 * math.pi * x @ b
        normals = torch.from_numpy(self.geometry[:, 3:]).to(self.dev)
        return torch.cat([torch.sin(proj), torch.cos(proj), normals], 1).to(self.dtype)

    def cond_features(self, conditions):
        c = np.asarray(conditions, np.float32)
        out = np.stack([(c[:, 0] - self.m_lo) / (self.m_hi - self.m_lo), c[:, 1] / 15.0, np.log2(c[:, 2]) / 2.0], 1)
        return torch.from_numpy(out.astype(np.float32)).to(self.dev)

    def fit(self, conditions, fields):
        torch.manual_seed(SEED)
        self.m_lo, self.m_hi = float(np.min(conditions[:, 0])), float(np.max(conditions[:, 0]))
        self.feats = self.point_features()
        self.mean = torch.from_numpy(np.asarray(fields, np.float32).mean(0)).to(self.dev)
        y = torch.from_numpy(np.asarray(fields, np.float32)).to(self.dev) - self.mean
        cond = self.cond_features(conditions)
        cdf = torch.cumsum(torch.from_numpy(self.w).to(self.dev), 0)
        cdf = cdf / cdf[-1]
        self.net = Field(self.feats.shape[1] + 3).to(self.dev)
        opt = torch.optim.AdamW(self.net.parameters(), lr=NF_LR)
        use_amp = self.dev.type == "cuda"
        scaler = torch.amp.GradScaler(enabled=use_amp)
        start, step, recent = time.time(), 0, []
        while (frac := (time.time() - start) / NF_SECONDS) < 1:
            lr = NF_LR * min(1.0, frac / NF_WARMUP) * 0.5 * (1 + math.cos(math.pi * frac))
            for group in opt.param_groups:
                group["lr"] = lr
            runs = torch.randint(len(y), (NF_RUNS, 1), device=self.dev)
            pts = torch.searchsorted(cdf, torch.rand(NF_RUNS, NF_POINTS, device=self.dev)).clamp_max(len(cdf) - 1)
            x = torch.cat([self.feats[pts], cond[runs].expand(-1, NF_POINTS, -1).to(self.dtype)], -1)
            with torch.autocast(self.dev.type, dtype=torch.float16, enabled=use_amp):
                pred = self.net(x)
            loss = (pred.float() - y[runs, pts]).abs().mean()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if step % 50 == 0:
                recent.append(loss.item())
            step += 1
            if step % 2000 == 0:
                log(f"NeuralField step {step} {time.time() - start:.0f}s lr {lr:.2e} weighted L1 {np.mean(recent):.5f}")
                recent = []
        log(f"NeuralField trained {step} steps in {time.time() - start:.0f}s")
        del y
        return self

    @torch.no_grad()
    def predict(self, conditions):
        self.net.eval()
        cond = self.cond_features(conditions).to(self.dtype)
        n = len(self.feats)
        out = np.empty((len(conditions), n), np.float32)
        for i in range(len(conditions)):
            parts = []
            for s in range(0, n, NF_CHUNK):
                f = self.feats[s:s + NF_CHUNK]
                x = torch.cat([f, cond[i].expand(len(f), -1)], 1)
                with torch.autocast(self.dev.type, dtype=torch.float16, enabled=self.dev.type == "cuda"):
                    parts.append(self.net(x).float())
            out[i] = (torch.cat(parts) + self.mean).cpu().numpy()
        self.net.train()
        return out


def build(arm, meta):
    return LineModel(meta, **(ARMS[arm] if isinstance(arm, str) else arm))


class Solver(CRMSolver):

    name = "submission"

    def get_model(self, meta):
        log(f"ARM {ARM}")
        return build(ARM, meta)

    def fit(self, model, conditions, fields):
        model.fit(conditions, fields)
