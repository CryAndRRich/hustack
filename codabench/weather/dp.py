import math

import numpy as np
import torch
from numba import cuda

from config import EARTH_R, NM
from geo import latlon
from rise import A_W, DEAD_DEG, K_A, K_H, K_S, K_W

F32 = np.float32
FKH, FKA, FAW, FKW, FKS = F32(K_H), F32(K_A), F32(A_W), F32(K_W), F32(K_S)
FDEAD = F32(math.radians(DEAD_DEG))
F0, F1, FHALF, F015, F2R = F32(0.0), F32(1.0), F32(0.5), F32(0.15), F32(2.0 * EARTH_R)
F3600, FMWH, FINF = F32(3600.0), F32(1e-3), F32(3.0e38)


@cuda.jit(device=True, inline=True)
def _power2(u, v10, swh, s, c, sh, ch, vs):
    ux = vs - (u * sh + v10 * ch)
    uy = v10 * sh - u * ch
    vr2 = ux * ux + uy * uy
    vr = math.sqrt(vr2)
    p = FKH * vs * vs * vs + FKA * vs * (vr * ux - vs * vs)
    th = abs(math.atan2(s * ch - c * sh, c * ch + s * sh))
    p += FAW * swh * swh * vs * math.sqrt(vs) * math.exp(-FKW * th * th * th)
    sail = F0
    a = math.atan2(abs(uy), ux) - FDEAD
    if a >= F0:
        sa = math.sin(a)
        sail = FKS * sa * (F1 + F015 * sa * sa) * vr2 * vs
    return p, sail


@cuda.jit
def _power_kernel(w, sh, ch, vs, out_w, out_n):
    i = cuda.grid(1)
    if i < out_w.shape[0]:
        p, sail = _power2(w[i, 0], w[i, 1], w[i, 2], w[i, 3], w[i, 4], sh[i], ch[i], vs[i])
        out_w[i] = max(p - sail, F0)
        out_n[i] = max(p, F0)


@cuda.jit
def _dp_kernel(vw, vn, vw_next, vn_next, back_w, back_n, nj, off_di, off_dj, xyz, east, north, w0, w1, flag, ok_node,
               edge_ok, dt_h, dmax, big):
    q = cuda.grid(1)
    if q >= vn_next.shape[0]:
        return
    if ok_node[q] == 0:
        vw_next[q] = FINF
        vn_next[q] = FINF
        back_w[q] = -1
        back_n[q] = -1
        return
    i = q // nj
    j = q - i * nj
    qx, qy, qz = xyz[q, 0], xyz[q, 1], xyz[q, 2]
    bw = FINF
    bn = FINF
    aw = -1
    an = -1
    scale = FHALF * dt_h * FMWH
    for o in range(off_di.shape[0]):
        i0 = i - off_di[o]
        j0 = j - off_dj[o]
        if i0 < 0 or j0 < 0 or j0 >= nj:
            continue
        p = i0 * nj + j0
        vpw = vw[p]
        vpn = vn[p]
        if (vpw >= bw and vpn >= bn) or edge_ok[p, o] == 0:
            continue
        px, py, pz = xyz[p, 0], xyz[p, 1], xyz[p, 2]
        dx, dy, dz = qx - px, qy - py, qz - pz
        chord = math.sqrt(dx * dx + dy * dy + dz * dz)
        dist = F2R * math.asin(min(chord * FHALF, F1))
        if dist > dmax:
            continue
        vs = dist / (dt_h * F3600)
        dot = px * qx + py * qy + pz * qz
        tx, ty, tz = qx - dot * px, qy - dot * py, qz - dot * pz
        de = tx * east[p, 0] + ty * east[p, 1] + tz * east[p, 2]
        dn = tx * north[p, 0] + ty * north[p, 1] + tz * north[p, 2]
        dl = math.sqrt(de * de + dn * dn)
        sh = F0
        ch = F1
        if dl > F0:
            sh = de / dl
            ch = dn / dl
        p0, s0 = _power2(w0[p, 0], w0[p, 1], w0[p, 2], w0[p, 3], w0[p, 4], sh, ch, vs)
        p1, s1 = _power2(w1[q, 0], w1[q, 1], w1[q, 2], w1[q, 3], w1[q, 4], sh, ch, vs)
        pen = F0
        if flag[p] != 0 or flag[q] != 0:
            pen = big
        cw = vpw + (max(p0 - s0, F0) + max(p1 - s1, F0)) * scale + pen
        cn = vpn + (max(p0, F0) + max(p1, F0)) * scale + pen
        if cw < bw:
            bw = cw
            aw = o
        if cn < bn:
            bn = cn
            an = o
    vw_next[q] = bw
    vn_next[q] = bn
    back_w[q] = aw
    back_n[q] = an


def _dev(t):
    return cuda.as_cuda_array(t.contiguous())


class Lattice:
    def __init__(self, frame, corr, n_alpha, dbeta_deg, beta_max_deg, stage_h, vmax_kn, device, land=None,
                 node_land_km=0.0, edge_land_km=0.0, box_margin=0.0, centre=None, slope_max=0.0, quiet=False):
        self.frame, self.corr, self.device, self.stage_h = frame, corr, device, stage_h
        self.alpha = np.linspace(0.0, frame.theta, n_alpha + 1)
        nb = int(round(beta_max_deg / dbeta_deg)) if dbeta_deg > 0 else 0
        self.beta = np.radians(np.arange(-nb, nb + 1) * dbeta_deg)
        self.ni, self.nj = n_alpha + 1, 2 * nb + 1
        self.src, self.dst = nb, (self.ni - 1) * self.nj + nb
        mid = np.zeros(self.ni)
        if centre is not None:
            mid = np.interp(self.alpha, np.maximum.accumulate(centre[0]), centre[1])
            mid[0] = mid[-1] = 0.0
        aa, bb = np.meshgrid(self.alpha, self.beta, indexing="ij")
        bb = bb + mid[:, None]
        self.ab = np.stack([aa.ravel(), bb.ravel()], 1)
        xyz = frame.xyz(aa.ravel(), bb.ravel())
        lat, lon = latlon(xyz)
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        rho = np.sqrt(x * x + y * y)
        east = np.stack([-y / rho, x / rho, 0 * x], 1)
        north = np.stack([-z * x / rho, -z * y / rho, rho], 1)
        f32 = dict(dtype=torch.float32, device=device)
        self.xyz, self.east, self.north = (torch.tensor(a, **f32) for a in (xyz, east, north))
        self.lat = torch.tensor(lat, dtype=torch.float64, device=device)
        self.lonn = torch.tensor(corr.norm_lon(lon), dtype=torch.float64, device=device)
        _, fy, fx = corr.frac(self.lat, self.lonn, self.lat, clip=False)
        self.fy, self.fx = fy, fx
        step_a = frame.theta / n_alpha * EARTH_R / NM
        step_b = math.radians(dbeta_deg) * EARTH_R / NM if nb else 1.0
        reach = vmax_kn * stage_h
        n_di = int(reach / step_a) + 2
        drift = np.zeros(n_di, dtype=np.int64)
        if centre is not None and nb:
            for di in range(1, n_di):
                d = np.abs(mid[di:] - mid[:-di]).max() / math.radians(dbeta_deg) if di < self.ni else 0.0
                drift[di] = math.ceil(min(d, slope_max * di * step_a / step_b))
        offs = [(di, dj) for di in range(n_di) for dj in range(-nb, nb + 1)
                if math.hypot(di * step_a, max(0, abs(dj) - drift[di]) * step_b) <= reach * 1.0001]
        assert len(offs) < 32768, f"{len(offs)} offsets do not fit int16 back-pointers"
        self.off = np.array(offs, dtype=np.int32)
        self.off_di = torch.tensor(self.off[:, 0], dtype=torch.int32, device=device)
        self.off_dj = torch.tensor(self.off[:, 1], dtype=torch.int32, device=device)
        self.vmax = vmax_kn * NM / 3600.0
        n = self.ni * self.nj
        ok = ((fy >= box_margin) & (fy <= corr.nlat - 1 - box_margin) & (fx >= box_margin)
              & (fx <= corr.nlon - 1 - box_margin))
        if land is not None:
            ok &= land.km(self.lat, self.lonn) >= node_land_km
        ok[self.src] = True
        ok[self.dst] = True
        self.ok_node = ok.to(torch.uint8)
        self.edge_ok = torch.ones((n, len(offs)), dtype=torch.uint8, device=device)
        if land is not None:
            ii = torch.arange(n, device=device) // self.nj
            jj = torch.arange(n, device=device) % self.nj
            for o, (di, dj) in enumerate(offs):
                i1, j1 = ii + di, jj + dj
                valid = (i1 < self.ni) & (j1 >= 0) & (j1 < self.nj)
                q = torch.where(valid, i1 * self.nj + j1, torch.zeros_like(i1))
                good = valid.clone()
                for f in (0.25, 0.5, 0.75):
                    pm = (1 - f) * self.xyz + f * self.xyz[q]
                    pm = pm / pm.norm(dim=1, keepdim=True)
                    la, lo = latlon(pm.double())
                    good &= land.km(la, corr.norm_lon(lo)) >= edge_land_km
                self.edge_ok[:, o] = good.to(torch.uint8)
        if not quiet:
            print(f"lattice {corr.name}: {self.ni}x{self.nj} nodes ({int(ok.sum())} usable), {len(offs)} offsets, "
                  f"step {step_a:.2f} x {step_b:.2f} nm, stage {stage_h} h", flush=True)

    def stages(self, dep_h, passage_h):
        taus = list(np.arange(dep_h, dep_h + passage_h, self.stage_h)) + [dep_h + passage_h]
        return np.array(taus, dtype=np.float64)

    def node_weather(self, win, taus):
        fy = self.fy.clamp(0, self.corr.nlat - 1)
        fx = self.fx.clamp(0, self.corr.nlon - 1)
        idx, w = win.spatial_taps(fy, fx)
        return torch.stack([win.slab(float(t), idx, w)[:, :5] for t in taus])

    def interval_flags(self, win, taus, hs_lim, tws_lim, dilate):
        fy = self.fy.clamp(0, self.corr.nlat - 1)
        fx = self.fx.clamp(0, self.corr.nlon - 1)
        gx = 2 * fx / (self.corr.nlon - 1) - 1
        gy = 2 * fy / (self.corr.nlat - 1) - 1
        grid = torch.stack([gx, gy], -1).reshape(1, 1, -1, 2).float()
        flags = []
        for k in range(len(taus) - 1):
            h0, h1 = int(math.floor(taus[k])) - win.a, int(math.ceil(taus[k + 1])) - win.a
            raw = win.raw[h0:h1 + 1]
            hs = raw[..., 2].amax(0)
            tws = torch.sqrt(raw[..., 0] ** 2 + raw[..., 1] ** 2).amax(0)
            env = torch.stack([hs, tws])[None]
            if dilate:
                env = torch.nn.functional.max_pool2d(env, 3, stride=1, padding=1)
            val = torch.nn.functional.grid_sample(env, grid, mode="bilinear", align_corners=True).reshape(2, -1)
            flags.append(((val[0] > hs_lim) | (val[1] > tws_lim)).to(torch.uint8))
        return torch.stack(flags)

    def band_mask(self, lo_deg, hi_deg, span):
        a = torch.tensor(self.ab[:, 0] / self.frame.theta, device=self.device)
        b = torch.tensor(np.degrees(self.ab[:, 1]), device=self.device)
        out = (a >= span[0]) & (a <= span[1]) & ((b < lo_deg) | (b > hi_deg))
        return torch.where(out, torch.zeros_like(self.ok_node), self.ok_node)

    def solve(self, weather, flags, taus, big, ok=None):
        n, K = self.ni * self.nj, len(taus) - 1
        v = torch.full((4, n), float(FINF), dtype=torch.float32, device=self.device)
        v[:2, self.src] = 0.0
        back = torch.empty((2, K, n), dtype=torch.int16, device=self.device)
        dv = [_dev(v[i]) for i in range(4)]
        dback = [_dev(back[0]), _dev(back[1])]
        args = [_dev(t) for t in (self.off_di, self.off_dj, self.xyz, self.east, self.north)]
        dw, dflag = _dev(weather), _dev(flags)
        dok, dedge = _dev(self.ok_node if ok is None else ok), _dev(self.edge_ok)
        torch.cuda.synchronize()
        blocks = (n + 127) // 128
        cur, nxt = (0, 1), (2, 3)
        for k in range(K):
            dt = F32(taus[k + 1] - taus[k])
            _dp_kernel[blocks, 128](dv[cur[0]], dv[cur[1]], dv[nxt[0]], dv[nxt[1]], dback[0][k], dback[1][k], self.nj,
                                    *args, dw[k], dw[k + 1], dflag[k], dok, dedge, dt,
                                    F32(self.vmax * dt * 3600.0 * 1.0001), F32(big))
            cur, nxt = nxt, cur
        cuda.synchronize()
        back = back.cpu().numpy()
        paths = {}
        for mode, row in ((1, cur[0]), (0, cur[1])):
            if not float(v[row, self.dst]) < 1e37:
                paths[mode] = None
                continue
            path, q = [self.dst], self.dst
            bk = back[0 if mode else 1]
            for k in range(K - 1, -1, -1):
                o = int(bk[k, q])
                assert o >= 0, f"broken back-pointer at stage {k}"
                q = (q // self.nj - self.off[o, 0]) * self.nj + (q % self.nj - self.off[o, 1])
                path.append(q)
            path.reverse()
            assert path[0] == self.src
            paths[mode] = np.array(path)
        return paths

    def waypoints(self, path, taus, per_h):
        ab = self.ab[path]
        out = [ab[:1]]
        for k in range(len(path) - 1):
            m = int(round((taus[k + 1] - taus[k]) * per_h))
            f = (np.arange(1, m + 1) / m)[:, None]
            out.append((1 - f) * ab[k] + f * ab[k + 1])
        return np.concatenate(out)


def reference_check(device):
    rng = np.random.default_rng(1)
    ni, nj, K = 6, 5, 4
    n = ni * nj
    lat = 40 + rng.uniform(-1, 1, n)
    lon = -30 + np.repeat(np.arange(ni), nj) * 0.3 + rng.uniform(-0.05, 0.05, n)
    la, lo = np.radians(lat), np.radians(lon)
    xyz = np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], 1)
    rho = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2)
    east = np.stack([-xyz[:, 1] / rho, xyz[:, 0] / rho, 0 * rho], 1)
    north = np.stack([-xyz[:, 2] * xyz[:, 0] / rho, -xyz[:, 2] * xyz[:, 1] / rho, rho], 1)
    w = rng.uniform(-1, 1, (K + 1, n, 5)) * np.array([15, 15, 0, 1, 1])
    w[..., 2] = rng.uniform(0, 6, (K + 1, n))
    offs = np.array([(0, 0), (1, 0), (1, 1), (1, -1), (2, 0), (0, 1), (0, -1)], dtype=np.int32)
    flags = (rng.uniform(size=(K, n)) < 0.1).astype(np.uint8)
    ok = np.ones(n, np.uint8)
    edge = (rng.uniform(size=(n, len(offs))) < 0.9).astype(np.uint8)
    dt, dmax, big = 3.0, 1e9, 1e4

    def power_np(wv, sh, ch, vs, wps):
        u, v10, swh, s, c = wv
        ux = vs - (u * sh + v10 * ch)
        uy = v10 * sh - u * ch
        vr2 = ux * ux + uy * uy
        p = K_H * vs ** 3 + K_A * vs * (math.sqrt(vr2) * ux - vs * vs)
        th = abs(math.atan2(s * ch - c * sh, c * ch + s * sh))
        p += A_W * swh * swh * vs ** 1.5 * math.exp(-K_W * th ** 3)
        a = math.atan2(abs(uy), ux) - math.radians(DEAD_DEG)
        if wps and a >= 0:
            p -= K_S * math.sin(a) * (1 + 0.15 * math.sin(a) ** 2) * vr2 * vs
        return max(p, 0.0)

    refs = {}
    for wps in (1, 0):
        vref = np.full(n, np.inf)
        vref[2] = 0.0
        for k in range(K):
            vnew = np.full(n, np.inf)
            for q in range(n):
                i, j = divmod(q, nj)
                for o, (di, dj) in enumerate(offs):
                    i0, j0 = i - di, j - dj
                    if i0 < 0 or j0 < 0 or j0 >= nj or not edge[i0 * nj + j0, o]:
                        continue
                    p = i0 * nj + j0
                    if not np.isfinite(vref[p]):
                        continue
                    dist = 2 * EARTH_R * math.asin(min(np.linalg.norm(xyz[q] - xyz[p]) / 2, 1))
                    t = xyz[q] - (xyz[p] @ xyz[q]) * xyz[p]
                    de, dn = t @ east[p], t @ north[p]
                    dl = math.hypot(de, dn)
                    sh, ch = (de / dl, dn / dl) if dl > 0 else (0.0, 1.0)
                    vs = dist / (dt * 3600)
                    cost = vref[p] + 0.5 * (power_np(w[k, p], sh, ch, vs, wps)
                                            + power_np(w[k + 1, q], sh, ch, vs, wps)) * dt / 1000
                    cost += big if (flags[k, p] or flags[k, q]) else 0.0
                    vnew[q] = min(vnew[q], cost)
            vref = vnew
        refs[wps] = vref
    f32 = dict(dtype=torch.float32, device=device)
    v = torch.full((4, n), float(FINF), **f32)
    v[:2, 2] = 0.0
    back = torch.empty((2, K, n), dtype=torch.int16, device=device)
    dv = [_dev(v[i]) for i in range(4)]
    tw = torch.tensor(w, **f32)
    tf = torch.tensor(flags, device=device)
    args = [_dev(torch.tensor(offs[:, 0], device=device)), _dev(torch.tensor(offs[:, 1], device=device))]
    args += [_dev(torch.tensor(a, **f32)) for a in (xyz, east, north)]
    dok, dedge = _dev(torch.tensor(ok, device=device)), _dev(torch.tensor(edge, device=device))
    torch.cuda.synchronize()
    cur, nxt = (0, 1), (2, 3)
    for k in range(K):
        _dp_kernel[1, 64](dv[cur[0]], dv[cur[1]], dv[nxt[0]], dv[nxt[1]], _dev(back[0, k]), _dev(back[1, k]), nj,
                          *args, _dev(tw[k]), _dev(tw[k + 1]), _dev(tf[k]), dok, dedge, F32(dt), F32(dmax), F32(big))
        cur, nxt = nxt, cur
    cuda.synchronize()
    for wps, row in ((1, cur[0]), (0, cur[1])):
        got = v[row].double().cpu().numpy()
        vref = refs[wps]
        fin = np.isfinite(vref)
        assert np.array_equal(fin, got < 1e37), "reachability differs"
        rel = np.max(np.abs(got[fin] - vref[fin]) / np.maximum(1.0, np.abs(vref[fin])))
        assert rel < 1e-4, f"DP kernel mismatch (wps {wps}) {rel}"
    print("dp reference check ok (both modes)", flush=True)
