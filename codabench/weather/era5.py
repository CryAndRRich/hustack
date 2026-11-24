import math
import os
import zipfile
from pathlib import Path

import h5py
import numpy as np
import scipy.ndimage as ndi
import torch

from config import T0

PAD = 12
TMARGIN = 32
LONG_TO_SHORT = {
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "significant_height_of_combined_wind_waves_and_swell": "swh",
    "mean_wave_direction": "mwd",
}
MARKER = "era5_wind_atlantic_2024.nc"


def find_era5_dir():
    zips = []
    for root, dirs, files in os.walk("/kaggle/input"):
        if MARKER in files and "ne_10m_land.shp" in files:
            return Path(root)
        zips += [Path(root) / f for f in files if f == "weather-era5.zip"]
        dirs[:] = [d for d in dirs if root.count(os.sep) < 6]
    assert zips, "dataset weather-era5 not attached (no era5 files and no weather-era5.zip under /kaggle/input)"
    target = Path("/tmp/era5")
    if not (target / MARKER).exists():
        print(f"extracting {zips[0]} -> {target}", flush=True)
        with zipfile.ZipFile(zips[0]) as zf:
            zf.extractall(target)
    return target


def _attr(ds, name, default=None):
    v = ds.attrs.get(name, default)
    if isinstance(v, bytes):
        return v.decode()
    if isinstance(v, np.ndarray) and v.size == 1:
        v = v.reshape(-1)[0]
        return v.decode() if isinstance(v, bytes) else v
    return v


def hours_since_t0(tv):
    unit, ref = _attr(tv, "units").split(" since ")
    step = {"seconds": 1 / 3600, "minutes": 1 / 60, "hours": 1.0, "days": 24.0}[unit.strip().lower()]
    ref = ref.strip().replace(" ", "T").rstrip("Z")
    ref = ref.split("+")[0]
    base = (np.datetime64(ref, "s") - np.datetime64(T0, "s")) / np.timedelta64(1, "h")
    return base + np.asarray(tv[:], dtype=np.float64) * step


class Corridor:
    def __init__(self, root, name):
        self.name = name
        self.ds = {kind: [h5py.File(root / f"era5_{kind}_{name}_{p}.nc", "r") for p in ("2024", "2025_01")]
                   for kind in ("wind", "waves")}
        ds0 = self.ds["wind"][0]
        lat = np.asarray(ds0["latitude"][:], dtype=np.float64)
        lon = np.asarray(ds0["longitude"][:], dtype=np.float64)
        self.flip = bool(lat[0] > lat[-1])
        self.lat = lat[::-1] if self.flip else lat
        assert np.all(np.diff(lon) > 0)
        self.lon = lon
        assert np.allclose(np.diff(self.lat), 0.25) and np.allclose(np.diff(self.lon), 0.25)
        self.names, self.ntimes = {}, {}
        hours = []
        for kind, dss in self.ds.items():
            self.ntimes[kind] = []
            for ds in dss:
                assert np.array_equal(np.asarray(ds["latitude"][:]), np.asarray(ds0["latitude"][:]))
                tname = "valid_time" if "valid_time" in ds else "time"
                h = hours_since_t0(ds[tname])
                if kind == "wind":
                    hours.append(h)
                self.ntimes[kind].append(len(h))
                for v, obj in ds.items():
                    if isinstance(obj, h5py.Dataset) and obj.ndim == 3:
                        self.names[(kind, LONG_TO_SHORT.get(v, v))] = v
        hours = np.concatenate(hours)
        assert self.ntimes["wind"] == self.ntimes["waves"], self.ntimes
        assert np.allclose(hours, np.arange(len(hours))), "ERA5 hours are not contiguous from 2024-01-01 00:00"
        self.T = len(hours)
        self.nlat, self.nlon = len(self.lat), len(self.lon)
        self.lon_shift = bool(self.lon[0] >= 0 and self.lon[-1] > 180)
        info = {k: (str(self.ds[k[0]][0][v].dtype), _attr(self.ds[k[0]][0][v], "scale_factor"),
                    _attr(self.ds[k[0]][0][v], "_FillValue")) for k, v in self.names.items()}
        print(f"{name}: T={self.T} lat {self.lat[0]}..{self.lat[-1]} ({self.nlat}) lon {self.lon[0]}..{self.lon[-1]} "
              f"({self.nlon}) vars {info}", flush=True)
        for short in ("u10", "v10", "swh", "mwd"):
            assert (("wind" if short[0] in "uv" else "waves"), short) in self.names, f"{short} missing"

    def read(self, kind, short, a, b):
        parts, off = [], 0
        for ds, n in zip(self.ds[kind], self.ntimes[kind]):
            lo, hi = max(a - off, 0), min(b - off, n)
            if lo < hi:
                var = ds[self.names[(kind, short)]]
                x = var[lo:hi]
                scale, offset = _attr(var, "scale_factor"), _attr(var, "add_offset")
                if scale is not None or offset is not None:
                    x = x * (1.0 if scale is None else scale) + (0.0 if offset is None else offset)
                parts.append(np.asarray(x, dtype=np.float32))
            off += n
        arr = np.concatenate(parts)
        np.nan_to_num(arr, copy=False, nan=0.0)
        if self.flip:
            arr = arr[:, ::-1, :]
        return np.ascontiguousarray(arr)

    def norm_lon(self, lon):
        if not self.lon_shift:
            return lon
        return torch.where(lon < 0, lon + 360.0, lon) if torch.is_tensor(lon) else np.where(lon < 0, lon + 360.0, lon)

    def frac(self, lat, lon_normed, t_h, clip=True):
        fy = (lat - self.lat[0]) / 0.25
        fx = (lon_normed - self.lon[0]) / 0.25
        if not clip:
            return t_h, fy, fx
        if torch.is_tensor(lat):
            return (torch.clamp(t_h, 0, self.T - 1), torch.clamp(fy, 0, self.nlat - 1),
                    torch.clamp(fx, 0, self.nlon - 1))
        return np.clip(t_h, 0, self.T - 1), np.clip(fy, 0, self.nlat - 1), np.clip(fx, 0, self.nlon - 1)


Z_POLE = math.sqrt(3.0) - 2.0


def read_pack(corr, h_lo, h_hi):
    a = max(0, int(np.floor(h_lo)) - TMARGIN)
    b = min(corr.T, int(np.ceil(h_hi)) + TMARGIN + 1)
    mwd = corr.read("waves", "mwd", a, b)
    rad = np.radians(mwd)
    arrs = [corr.read("wind", "u10", a, b), corr.read("wind", "v10", a, b), corr.read("waves", "swh", a, b),
            np.sin(rad).astype(np.float32), np.cos(rad).astype(np.float32), mwd]
    return a, arrs


def _filter_axis0(c):
    z = Z_POLE
    n = c.shape[0]
    x_last = c[n - 1].clone()
    c[0] = c[0] / (1 - z)
    for i in range(1, n):
        c[i] += z * c[i - 1]
    big = x_last / (1 - z)
    c_last = c[n - 1].clone()
    c[n - 1] = -(z * big / (1 - z) + z * (c_last - big) / (1 - z * z))
    for i in range(n - 2, -1, -1):
        c[i] = z * (c[i + 1] - c[i])
    c *= 6.0


def spline_coef(x):
    c = torch.nn.functional.pad(x[None, None], (PAD,) * 6, mode="replicate")[0, 0]
    for dim in range(3):
        m = c.movedim(dim, 0).contiguous()
        _filter_axis0(m)
        c = m.movedim(0, dim)
    return c.contiguous()


def bspline_w(u):
    u2 = u * u
    u3 = u2 * u
    return [(1 - u) ** 3 / 6.0, (3 * u3 - 6 * u2 + 4) / 6.0, (-3 * u3 + 3 * u2 + 3 * u + 1) / 6.0, u3 / 6.0]


class Window:
    def __init__(self, corr, pack, device):
        self.corr = corr
        self.a, arrs = pack
        nt = arrs[0].shape[0]
        self.raw = torch.empty((nt, corr.nlat, corr.nlon, 6), dtype=torch.float32, device=device)
        self.coef = torch.empty((nt + 2 * PAD, corr.nlat + 2 * PAD, corr.nlon + 2 * PAD, 6), dtype=torch.float32,
                                device=device)
        for c, arr in enumerate(arrs):
            x = torch.from_numpy(arr).to(device)
            self.raw[..., c] = x
            self.coef[..., c] = spline_coef(x)
        self.Tp, self.Yp, self.Xp = self.coef.shape[:3]
        self.flat = self.coef.view(-1, 6)
        r = torch.arange(4, device=device)
        self.taps = ((r[:, None, None] * self.Yp + r[None, :, None]) * self.Xp + r[None, None, :]).reshape(64)

    def cubic(self, ft, fy, fx):
        t = ft - self.a + PAD
        y = fy + PAD
        x = fx + PAD
        bt, by, bx = torch.floor(t), torch.floor(y), torch.floor(x)
        wt, wy, wx = (torch.stack(bspline_w(q), -1) for q in (t - bt, y - by, x - bx))
        w = (wt[..., :, None, None] * wy[..., None, :, None] * wx[..., None, None, :]).flatten(-3)
        base = ((bt.long() - 1) * self.Yp + (by.long() - 1)) * self.Xp + (bx.long() - 1)
        vals = self.flat[base[..., None] + self.taps].to(ft.dtype)
        return (w[..., None] * vals).sum(-2)

    def linear(self, ft, fy, fx):
        corr = self.corr
        i0t = torch.clamp(torch.floor(ft).long(), 0, corr.T - 2)
        i0y = torch.clamp(torch.floor(fy).long(), 0, corr.nlat - 2)
        i0x = torch.clamp(torch.floor(fx).long(), 0, corr.nlon - 2)
        wt, wy, wx = (ft - i0t).float(), (fy - i0y).float(), (fx - i0x).float()
        out = torch.zeros(ft.shape + (6,), dtype=torch.float32, device=ft.device)
        for dt in (0, 1):
            for dy in (0, 1):
                for dx in (0, 1):
                    w = (wt if dt else 1 - wt) * (wy if dy else 1 - wy) * (wx if dx else 1 - wx)
                    it = torch.clamp(i0t + dt, 0, corr.T - 1) - self.a
                    out = out + w.unsqueeze(-1) * self.raw[it, torch.clamp(i0y + dy, 0, corr.nlat - 1),
                                                           torch.clamp(i0x + dx, 0, corr.nlon - 1)]
        return out.to(ft.dtype)

    def spatial_taps(self, fy, fx):
        y, x = fy + PAD, fx + PAD
        by, bx = torch.floor(y), torch.floor(x)
        wy, wx = torch.stack(bspline_w(y - by), 1), torch.stack(bspline_w(x - bx), 1)
        r = torch.arange(4, device=fy.device)
        iy = by.long()[:, None] - 1 + r
        ix = bx.long()[:, None] - 1 + r
        idx = (iy[:, :, None] * self.Xp + ix[:, None, :]).reshape(-1, 16)
        w = (wy[:, :, None] * wx[:, None, :]).reshape(-1, 16).float()
        return idx, w

    def slab(self, tau, idx, w):
        t = tau - self.a + PAD
        bt = int(np.floor(t))
        wt = bspline_w(t - bt)
        sl = sum(float(wt[m]) * self.coef[bt - 1 + m] for m in range(4)).view(-1, 6)
        return (sl[idx] * w[..., None]).sum(1)

    def check_against_scipy(self, n=300, seed=0):
        rng = np.random.default_rng(seed)
        nt = self.raw.shape[0]
        lo, hi = (TMARGIN, nt - TMARGIN - 1) if nt > 2 * TMARGIN + 2 else (0, nt - 1)
        ft = rng.uniform(lo, hi, n) + self.a
        fy = rng.uniform(0, self.corr.nlat - 1, n)
        fx = rng.uniform(0, self.corr.nlon - 1, n)
        dev = self.coef.device
        ours = self.cubic(*(torch.tensor(a, dtype=torch.float64, device=dev) for a in (ft, fy, fx))).cpu().numpy()
        errs = []
        for c in range(6):
            arr = self.raw[..., c].cpu().numpy()
            ref = ndi.map_coordinates(arr, np.array([ft - self.a, fy, fx]), order=3, mode="nearest")
            errs.append(float(np.max(np.abs(ref - ours[:, c]))))
        return errs
