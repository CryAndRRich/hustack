import numpy as np
from matplotlib.path import Path as Polygon
from numba import njit, prange
from scipy.spatial import cKDTree

C = 299792458.0
WEEK = 604800.0
GPS_EPOCH = np.datetime64("1980-01-06T00:00:00")
LEAP, BDT_LAG, BDS_WEEK0 = 18.0, 14.0, 1356
MU = {"G": 3.986005e14, "E": 3.986004418e14, "C": 3.986004418e14}
WE = {"G": 7.2921151467e-5, "E": 7.2921151467e-5, "C": 7.292115e-5, "R": 7.292115e-5}
FREQ = {"G": 1575.42e6, "E": 1575.42e6, "C": 1561.098e6}
A_WGS, F_WGS = 6378137.0, 1 / 298.257223563
E2 = F_WGS * (2 - F_WGS)
LAT0, LON0 = 22.31, 114.19
OPEN = ("Open-sided Structure", "Temporary Structure")


def num(s):
    s = s.strip().replace("D", "E")
    return float(s) if s else np.nan


def gps_seconds(line):
    y, mo, d, h, mi, s = (int(line[4:8]), int(line[9:11]), int(line[12:14]), int(line[15:17]), int(line[18:20]),
                          int(line[21:23]))
    return (np.datetime64(f"{y:04d}-{mo:02d}-{d:02d}T{h:02d}:{mi:02d}:00") - GPS_EPOCH) / np.timedelta64(1, "s") + s


class Ephemeris:
    def __init__(self, paths):
        self.iono, recs = {}, {}
        for path in paths:
            lines = open(path).read().splitlines()
            i = next(k for k, x in enumerate(lines) if "END OF HEADER" in x) + 1
            for x in lines[:i]:
                if "IONOSPHERIC CORR" in x:
                    self.iono[x[:4].strip()] = [num(x[5 + 12 * j:17 + 12 * j]) for j in range(4)]
            while i < len(lines):
                head, j = lines[i], i + 1
                while j < len(lines) and lines[j].startswith("    "):
                    j += 1
                vals = [num(head[23 + 19 * k:42 + 19 * k]) for k in range(3)]
                for x in lines[i + 1:j]:
                    vals += [num(x[4 + 19 * k:23 + 19 * k]) for k in range(4)]
                i = j
                sysc, prn = head[0], int(head[1:3])
                toc = gps_seconds(head)
                if sysc == "R":
                    if len(vals) < 15 or vals[6] != 0:
                        continue
                    toc += LEAP
                    toe = toc
                else:
                    if len(vals) < 25 or vals[24] != 0 or (sysc == "C" and (prn <= 5 or prn >= 59)):
                        continue
                    if sysc == "E" and not int(vals[20]) & 1:
                        continue
                    if sysc == "C":
                        toc += BDT_LAG
                        toe = (vals[21] + BDS_WEEK0) * WEEK + vals[11] + BDT_LAG
                    else:
                        toe = vals[21] * WEEK + vals[11]
                recs.setdefault(f"{sysc}{prn}", {})[toe] = (toc, vals[:29] + [np.nan] * (29 - len(vals[:29])))
        self.recs = {}
        for sat, d in recs.items():
            toes = np.array(sorted(d))
            self.recs[sat] = (toes, np.array([d[t][0] for t in toes]), np.array([d[t][1] for t in toes]))

    def has(self, sat):
        return sat in self.recs

    def glonass_freq(self, sat, t):
        toes, _, p = self.recs[sat]
        return 1602e6 + p[np.abs(toes - t).argmin(), 10] * 562500.0

    def state(self, sat, t):
        toes, tocs, p = self.recs[sat]
        k = np.abs(t[:, None] - toes[None, :]).argmin(1)
        pk, toe, toc = p[k], toes[k], tocs[k]
        if sat[0] == "R":
            pos, vel = glonass_orbit(pk, t - toe)
            clk = pk[:, 0] + pk[:, 1] * (t - toc)
            return pos, vel, clk, pk[:, 1]
        pos, rel = kepler(pk, t - toe, sat[0])
        pos2, _ = kepler(pk, t + 0.5 - toe, sat[0])
        pos1, _ = kepler(pk, t - 0.5 - toe, sat[0])
        dt = t - toc
        tgd = pk[:, 26] if sat[0] == "E" else pk[:, 25]
        clk = pk[:, 0] + pk[:, 1] * dt + pk[:, 2] * dt ** 2 + rel - tgd
        return pos, pos2 - pos1, clk, pk[:, 1] + 2 * pk[:, 2] * dt


def kepler(p, tk, sysc):
    mu, we = MU[sysc], WE[sysc]
    a = p[:, 10] ** 2
    e = p[:, 8]
    m = p[:, 6] + (np.sqrt(mu / a ** 3) + p[:, 5]) * tk
    ea = m.copy()
    for _ in range(12):
        ea = m + e * np.sin(ea)
    nu = np.arctan2(np.sqrt(1 - e ** 2) * np.sin(ea), np.cos(ea) - e)
    phi = nu + p[:, 17]
    s2, c2 = np.sin(2 * phi), np.cos(2 * phi)
    u = phi + p[:, 9] * s2 + p[:, 7] * c2
    r = a * (1 - e * np.cos(ea)) + p[:, 4] * s2 + p[:, 16] * c2
    inc = p[:, 15] + p[:, 14] * s2 + p[:, 12] * c2 + p[:, 19] * tk
    om = p[:, 13] + (p[:, 18] - we) * tk - we * p[:, 11]
    xp, yp = r * np.cos(u), r * np.sin(u)
    pos = np.stack([xp * np.cos(om) - yp * np.cos(inc) * np.sin(om),
                    xp * np.sin(om) + yp * np.cos(inc) * np.cos(om),
                    yp * np.sin(inc)], 1)
    rel = -2 * np.sqrt(mu) / C ** 2 * e * p[:, 10] * np.sin(ea)
    return pos, rel


def glonass_orbit(p, dt):
    mu, j2, ae, we = 3.9860044e14, 1.08263e-3, 6378136.0, WE["R"]
    y = np.stack([p[:, 3], p[:, 7], p[:, 11], p[:, 4], p[:, 8], p[:, 12]], 1) * 1e3
    acc = np.stack([p[:, 5], p[:, 9], p[:, 13]], 1) * 1e3

    def deriv(s):
        r2 = (s[:, :3] ** 2).sum(1)
        r3, z2 = r2 * np.sqrt(r2), 5 * s[:, 2] ** 2 / r2
        c = 1.5 * j2 * mu * ae ** 2 / (r2 * r3)
        ax = -mu * s[:, 0] / r3 - c * s[:, 0] * (1 - z2) + we ** 2 * s[:, 0] + 2 * we * s[:, 4] + acc[:, 0]
        ay = -mu * s[:, 1] / r3 - c * s[:, 1] * (1 - z2) + we ** 2 * s[:, 1] - 2 * we * s[:, 3] + acc[:, 1]
        az = -mu * s[:, 2] / r3 - c * s[:, 2] * (3 - z2) + acc[:, 2]
        return np.concatenate([s[:, 3:], np.stack([ax, ay, az], 1)], 1)

    n = max(1, int(np.ceil(np.abs(dt).max() / 30.0)))
    h = (dt / n)[:, None]
    for _ in range(n):
        k1 = deriv(y)
        k2 = deriv(y + h / 2 * k1)
        k3 = deriv(y + h / 2 * k2)
        k4 = deriv(y + h * k3)
        y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return y[:, :3], y[:, 3:]


def geodetic_to_ecef(lat, lon, h):
    lat, lon = np.radians(lat), np.radians(lon)
    n = A_WGS / np.sqrt(1 - E2 * np.sin(lat) ** 2)
    return np.stack([(n + h) * np.cos(lat) * np.cos(lon), (n + h) * np.cos(lat) * np.sin(lon),
                     (n * (1 - E2) + h) * np.sin(lat)], -1)


def ecef_to_geodetic(xyz):
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1 - E2))
    for _ in range(6):
        n = A_WGS / np.sqrt(1 - E2 * np.sin(lat) ** 2)
        h = p / np.cos(lat) - n
        lat = np.arctan2(z, p * (1 - E2 * n / (n + h)))
    return np.degrees(lat), np.degrees(np.arctan2(y, x)), h


def enu_rot(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.array([[-np.sin(lo), np.cos(lo), 0],
                     [-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)],
                     [np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]])


ROT0 = enu_rot(LAT0, LON0)
_S = np.sin(np.radians(LAT0))
M0 = A_WGS * (1 - E2) / (1 - E2 * _S ** 2) ** 1.5
N0 = A_WGS / np.sqrt(1 - E2 * _S ** 2)


def to_map(lat, lon):
    return (np.radians(lon - LON0) * N0 * np.cos(np.radians(LAT0)), np.radians(lat - LAT0) * M0)


def klobuchar(ab, lat, lon, az, el, tow):
    alpha, beta = ab
    el_s = el / np.pi
    psi = 0.0137 / (el_s + 0.11) - 0.022
    phi = np.clip(lat / 180 + psi * np.cos(az), -0.416, 0.416)
    lam = lon / 180 + psi * np.sin(az) / np.cos(phi * np.pi)
    phm = phi + 0.064 * np.cos((lam - 1.617) * np.pi)
    t = np.mod(4.32e4 * lam + tow, 86400.0)
    f = 1 + 16 * (0.53 - el_s) ** 3
    amp = np.maximum(sum(alpha[i] * phm ** i for i in range(4)), 0)
    per = np.maximum(sum(beta[i] * phm ** i for i in range(4)), 72000.0)
    x = 2 * np.pi * (t - 50400) / per
    return C * f * np.where(np.abs(x) < 1.57, 5e-9 + amp * (1 - x ** 2 / 2 + x ** 4 / 24), 5e-9)


def sat_states(eph, sats, t_rx, code):
    n = len(sats)
    pos, vel = np.full((n, 3), np.nan), np.full((n, 3), np.nan)
    clk, drift, lam = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    for s in np.unique(sats):
        if not eph.has(s):
            continue
        m = sats == s
        t = t_rx[m] - code[m] / C
        _, _, ck, _ = eph.state(s, t)
        t = t - ck
        p, v, ck, dr = eph.state(s, t)
        ang = WE[s[0]] * code[m] / C
        ca, sa = np.cos(ang), np.sin(ang)
        pos[m] = np.stack([ca * p[:, 0] + sa * p[:, 1], -sa * p[:, 0] + ca * p[:, 1], p[:, 2]], 1)
        vel[m] = np.stack([ca * v[:, 0] + sa * v[:, 1], -sa * v[:, 0] + ca * v[:, 1], v[:, 2]], 1)
        clk[m], drift[m] = ck, dr
        lam[m] = C / (eph.glonass_freq(s, t.mean()) if s[0] == "R" else FREQ[s[0]])
    return pos, vel, clk, drift, lam


def los_geometry(rx, sp):
    d = sp - rx
    rng = np.linalg.norm(d, axis=-1)
    u = d / rng[..., None]
    lat, lon, _ = ecef_to_geodetic(rx)
    enu = u @ enu_rot(lat, lon).T
    return rng, u, np.arcsin(np.clip(enu[..., 2], -1, 1)), np.mod(np.arctan2(enu[..., 0], enu[..., 1]), 2 * np.pi)


def huber(r, k=1.5):
    a = np.abs(r)
    return np.where(a <= k, 1.0, k / np.maximum(a, 1e-9))


def obs_sigma(el, cn0):
    return 3.0 / np.maximum(np.sin(el), 0.1) * np.sqrt(np.clip(10 ** ((40 - cn0) / 10), 0.3, 30))


def wls(h, r, sig, iters=6):
    w = np.ones(len(r))
    for _ in range(iters):
        wt = w / sig ** 2
        n = h.T @ (h * wt[:, None]) + 1e-6 * np.eye(h.shape[1])
        dx = np.linalg.solve(n, h.T @ (wt * r))
        res = r - h @ dx
        w = huber(res / sig)
    return dx, res, np.linalg.inv(n), w


def solve_epoch(rx, sp, clk, code, sysv, iono, tow, isb, systems):
    lat, lon, _ = ecef_to_geodetic(rx)
    rng, u, el, az = los_geometry(rx, sp)
    tro = 2.47 / (np.sin(np.maximum(el, 0.05)) + 0.0121)
    ion = klobuchar(iono, lat, lon, az, np.maximum(el, 0.05), tow) if iono is not None else 0.0
    pred = rng - C * clk + tro + ion + np.array([isb.get(s, 0.0) for s in sysv])
    cols = [np.ones(len(sysv))] if systems == ["*"] else [(sysv == s).astype(float) for s in systems]
    return np.column_stack([-u] + cols), code - pred, el, az


def spp(t, sp, clk, sysv, code, cn0, iono, isb, x0):
    good = np.isfinite(sp[:, 0])
    systems = ["*"] if isb else sorted(set(sysv[good]))
    if good.sum() < 4 + len(systems):
        return None
    sp, clk, code, sysv, cn0 = sp[good], clk[good], code[good], sysv[good], cn0[good]
    rx, bias, tow = x0.copy(), None, np.mod(t, WEEK)
    for it in range(8):
        h, r, el, az = solve_epoch(rx, sp, clk, code, sysv, iono, tow, isb, systems)
        if bias is None:
            bias = np.array([np.median(r[h[:, 3 + j] > 0])
                             for j in range(len(systems))])
        r = r - h[:, 3:] @ bias
        ok = el > np.radians(5) if it > 1 else np.ones(len(el), bool)
        if ok.sum() < 4 + len(systems):
            return None
        sig = obs_sigma(el, cn0)
        dx, res, cov, w = wls(h[ok], r[ok], sig[ok])
        rx, bias = rx + dx[:3], bias + dx[3:]
        if np.linalg.norm(dx[:3]) < 1e-3 and it > 2:
            break
    res_all = r - h @ dx
    hw = h[ok] / sig[ok, None] * np.sqrt(w)[:, None]
    lev = np.zeros(len(r))
    lev[ok] = np.einsum("ij,jk,ik->i", hw, cov, hw)
    out = {k: np.full(len(good), np.nan) for k in ("res", "nres", "loo", "el", "az")}
    out["res"][good], out["nres"][good] = res_all, res_all / sig
    out["loo"][good] = res_all / np.sqrt(np.clip(1 - lev, 0.05, 1)) / sig
    out["el"][good], out["az"][good] = el, az
    s2 = np.sum(w * (res / sig[ok]) ** 2) / max(ok.sum() - h.shape[1], 1)
    out.update(rx=rx, cov=cov[:3, :3] * max(s2, 1.0), bias=dict(zip(systems, bias)), n=int(ok.sum()))
    return out


def doppler_velocity(sp, sv, drift, lam, dop, cn0, rx, sign):
    good = np.isfinite(sp[:, 0]) & (dop != 0)
    out = np.full(len(dop), np.nan)
    if good.sum() < 5:
        return None, out
    _, u, el, _ = los_geometry(rx, sp[good])
    rr = sign * -lam[good] * dop[good]
    pred = (u * sv[good]).sum(1) - C * drift[good]
    h = np.column_stack([-u, np.ones(good.sum())])
    sig = 0.3 / np.maximum(np.sin(el), 0.1) * np.sqrt(np.clip(10 ** ((40 - cn0[good]) / 10), 0.3, 30))
    ok = el > np.radians(5)
    if ok.sum() < 5:
        return None, out
    dx, _, _, _ = wls(h[ok], (rr - pred)[ok], sig[ok])
    out[good] = (rr - pred - h @ dx) / sig
    return dx[:3], out


def rts_1d(z, r, v, dt, q):
    n = len(z)
    m, p, mp, pp = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    for k in range(n):
        if k == 0 or not np.isfinite(m[k - 1]):
            mp[k], pp[k] = (z[k], r[k] * 4) if np.isfinite(z[k]) else (np.nan, np.nan)
        else:
            step = 0.5 * (v[k - 1] + v[k]) * dt[k] if np.isfinite(v[k - 1] + v[k]) else 0.0
            qk = q * dt[k] if np.isfinite(v[k - 1] + v[k]) else 25.0 * dt[k] ** 2
            mp[k], pp[k] = m[k - 1] + step, p[k - 1] + qk
        m[k], p[k] = mp[k], pp[k]
        if np.isfinite(z[k]) and np.isfinite(mp[k]):
            innov = z[k] - mp[k]
            rk = r[k] * max(1.0, (innov / (4 * np.sqrt(pp[k] + r[k]))) ** 2)
            g = pp[k] / (pp[k] + rk)
            m[k], p[k] = mp[k] + g * innov, (1 - g) * pp[k]
    for k in range(n - 2, -1, -1):
        if np.isfinite(m[k]) and np.isfinite(mp[k + 1]) and pp[k + 1] > 0:
            g = p[k] / pp[k + 1]
            m[k] = m[k] + g * (m[k + 1] - mp[k + 1])
            p[k] = p[k] + g ** 2 * (p[k + 1] - pp[k + 1])
    return m


class City:
    def __init__(self, bld, max_seg=40.0):
        x1, y1, x2, y2, top = [], [], [], [], []
        self.polys = []
        for rings, t, solid in zip(bld.rings.values, bld.top_mpd.values, ~bld.block_type.isin(OPEN).values):
            for r, ring in enumerate(rings.split("|")):
                ll = np.array([[float(v) for v in pt.split()] for pt in ring.split(";")])
                x, y = to_map(ll[:, 1], ll[:, 0])
                if solid and r == 0:
                    self.polys.append(np.column_stack([x, y]))
                for k in range(len(x) - 1):
                    n = max(1, int(np.ceil(np.hypot(x[k + 1] - x[k], y[k + 1] - y[k]) / max_seg)))
                    f = np.linspace(0, 1, n + 1)
                    xs, ys = x[k] + f * (x[k + 1] - x[k]), y[k] + f * (y[k + 1] - y[k])
                    x1 += list(xs[:-1]); y1 += list(ys[:-1]); x2 += list(xs[1:]); y2 += list(ys[1:])
                    top += [t] * n
        self.seg = [np.array(a, dtype=np.float64) for a in (x1, y1, x2, y2, top)]
        self.half = max_seg / 2
        self.tree = cKDTree(np.column_stack([(self.seg[0] + self.seg[2]) / 2, (self.seg[1] + self.seg[3]) / 2]))
        cx = np.array([np.mean([float(pt.split()[0]) for pt in r.split("|")[0].split(";")]) for r in bld.rings])
        cy = np.array([np.mean([float(pt.split()[1]) for pt in r.split("|")[0].split(";")]) for r in bld.rings])
        bx, by = to_map(cy, cx)
        base = bld.base_mpd.values.astype(float)
        keep = np.isfinite(base)
        self.base_tree, self.base = cKDTree(np.column_stack([bx[keep], by[keep]])), base[keep]

    def inside(self, x, y):
        out = np.zeros(len(x), bool)
        tree = cKDTree(np.column_stack([x, y]))
        for poly in self.polys:
            c = poly.mean(0)
            idx = tree.query_ball_point(c, np.hypot(*(poly - c).T).max())
            if idx:
                out[idx] |= Polygon(poly).contains_points(np.column_stack([x[idx], y[idx]]))
        return out

    def ground(self, x, y, radius=40.0):
        out = np.full(len(x), np.nan)
        for i, idx in enumerate(self.base_tree.query_ball_point(np.column_stack([x, y]), radius)):
            if idx:
                out[i] = np.percentile(self.base[idx], 10)
        fill = np.nanmedian(out) if np.isfinite(out).any() else 5.0
        return np.where(np.isfinite(out), out, fill)

    def masks_at(self, x, y, h_ant, radius=500.0, block=64.0):
        out = np.full((len(x), 360), np.nan, dtype=np.float32)
        ii = np.flatnonzero(np.isfinite(x) & np.isfinite(y))
        if len(ii) == 0:
            return out
        xi, yi = x[ii], y[ii]
        h = self.ground(xi, yi) + h_ant
        key = np.floor(xi / block).astype(np.int64) * 1_000_000 + np.floor(yi / block).astype(np.int64)
        order = np.argsort(key, kind="stable")
        cuts = np.flatnonzero(np.diff(key[order])) + 1
        for sel in np.split(order, cuts):
            cx, cy = xi[sel].mean(), yi[sel].mean()
            reach = np.hypot(xi[sel] - cx, yi[sel] - cy).max()
            idx = np.array(self.tree.query_ball_point([cx, cy], radius + reach + self.half), dtype=np.int64)
            out[ii[sel]] = mask_batch(xi[sel], yi[sel], h[sel], *self.seg, idx, 360, self.half)
        return out


@njit(cache=False)
def mask_kernel(px, py, h, sx1, sy1, sx2, sy2, stop, idx, nbins, half):
    mask = np.zeros(nbins)
    bw = 2 * np.pi / nbins
    tmin = np.tan(np.radians(5.0))
    for k in idx:
        dh = stop[k] - h
        if not dh > 0:
            continue
        ax, ay, bx, by = sx1[k] - px, sy1[k] - py, sx2[k] - px, sy2[k] - py
        dmid = np.hypot(0.5 * (ax + bx), 0.5 * (ay + by))
        if dh / max(dmid - half, 1.0) < tmin:
            continue
        aa, ab = np.arctan2(ax, ay), np.arctan2(bx, by)
        d = ab - aa
        if d > np.pi:
            d -= 2 * np.pi
        elif d < -np.pi:
            d += 2 * np.pi
        n = int(abs(d) / bw) + 2
        ex, ey = bx - ax, by - ay
        for j in range(n):
            th = aa + d * j / (n - 1)
            ux, uy = np.sin(th), np.cos(th)
            den = ux * ey - uy * ex
            if abs(den) < 1e-12:
                t = min(np.hypot(ax, ay), np.hypot(bx, by))
            else:
                t = (ax * ey - ay * ex) / den
            if t < 0.5:
                t = 0.5
            el = np.arctan(dh / t)
            b = int((th % (2 * np.pi)) / bw) % nbins
            if el > mask[b]:
                mask[b] = el
    return mask


@njit(parallel=True, cache=False)
def mask_batch(px, py, h, sx1, sy1, sx2, sy2, stop, idx, nbins, half):
    out = np.zeros((len(px), nbins))
    for i in prange(len(px)):
        out[i] = mask_kernel(px[i], py[i], h[i], sx1, sy1, sx2, sy2, stop, idx, nbins, half)
    return out
