import json
import ssl
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import certifi
import numpy as np
import pandas as pd
from numcodecs import Blosc

from build_eod import RH_LEVEL
from build_nwp import SITES, window_starts

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "external" / "hres_points"
OUT = ROOT / "data" / "external" / "hres_daily.csv"
URL = "https://storage.googleapis.com/weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr"
URL15 = "https://storage.googleapis.com/weatherbench2/datasets/hres/2016-2022-0012-240x121_equiangular_with_poles_conservative.zarr"
LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
FIRST, LAST = pd.Timestamp("2016-01-01"), pd.Timestamp("2023-01-10")
VARS = {"2m_temperature": -273.15, "10m_wind_speed": None}
SAMPLES = [0, 6, 12, 18]
CTX = ssl.create_default_context(cafile=certifi.where())
WORKERS = 16


def site_weights():
    idx, wts = [], []
    for lat, lon in SITES.values():
        y, x = (lat + 90) / 0.25, lon / 0.25
        y0, x0 = int(y), int(x)
        fy, fx = y - y0, x - x0
        idx.append([(y0, x0), (y0, x0 + 1), (y0 + 1, x0), (y0 + 1, x0 + 1)])
        wts.append([(1 - fy) * (1 - fx), (1 - fy) * fx, fy * (1 - fx), fy * fx])
    return np.array(idx), np.array(wts)


IDX, WTS = site_weights()


def fetch(key):
    for attempt in range(5):
        try:
            with urllib.request.urlopen(key, timeout=120, context=CTX) as r:
                return Blosc().decode(r.read())
        except Exception as e:
            print(f"  retry {key}: {e}")
    raise RuntimeError(key)


def rh_points(init, block):
    path = RAW / f"rh15_{init:%Y%m%d%H}_{block}.json"
    if path.exists():
        return json.loads(path.read_text())
    t = (init - FIRST) // pd.Timedelta(hours=12)
    q, T = (np.frombuffer(fetch(f"{URL15}/{v}/{t}.{block}.0.0.0"), "<f4").reshape(8, 13, 240, 121)
            for v in ("specific_humidity", "temperature"))
    out = {}
    for j in range(8):
        vals = []
        for code, (lat, lon) in SITES.items():
            lev = RH_LEVEL.get(code, 1000)
            k = LEVELS.index(lev)
            x, y = lon / 1.5, (lat + 90) / 1.5
            x0, y0 = int(x), int(y)
            fx, fy = x - x0, y - y0
            w = np.array([(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy])
            pts = [(x0, y0), (x0 + 1, y0), (x0, y0 + 1), (x0 + 1, y0 + 1)]
            qq = sum(wi * q[j, k, a, b] for wi, (a, b) in zip(w, pts))
            tt = sum(wi * T[j, k, a, b] for wi, (a, b) in zip(w, pts)) - 273.15
            e = qq * lev / (0.622 + 0.378 * qq)
            vals.append(float(np.clip(100 * e / (6.112 * np.exp(17.67 * tt / (tt + 243.5))), 0, 100)))
        out[str(6 * (8 * block + j))] = vals
    path.write_text(json.dumps(out))
    return out


def points(var, init, hours):
    path = RAW / f"{var}_{init:%Y%m%d%H}_{hours:03d}.json"
    if path.exists():
        return json.loads(path.read_text())
    key = f"{URL}/{var}/{(init - FIRST) // pd.Timedelta(hours=12)}.{hours // 6}.0.0"
    field = np.frombuffer(fetch(key), "<f4").reshape(721, 1440)
    vals = (field[IDX[..., 0], IDX[..., 1]] * WTS).sum(1).tolist()
    path.write_text(json.dumps(vals))
    return vals


def inits():
    out = []
    for s in window_starts():
        day0 = s + pd.Timedelta(days=9)
        out += [d for d in (day0, day0 - pd.Timedelta(days=5)) if FIRST <= d and d + pd.Timedelta(days=6) <= LAST]
    return sorted(set(out))


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    jobs = [(v, i, 24 * L + h) for i in inits() for v in VARS for L in range(1, 6) for h in SAMPLES]
    print(f"{len(inits())} inits, {len(jobs)} chunks")
    with ThreadPoolExecutor(WORKERS) as ex:
        vals = list(ex.map(lambda j: points(*j), jobs))
    got = dict(zip(jobs, vals))
    blocks = [(i, b) for i in inits() for b in range(3)]
    with ThreadPoolExecutor(WORKERS // 2) as ex:
        rh = {}
        for (i, b), r in zip(blocks, ex.map(lambda j: rh_points(*j), blocks)):
            rh.update({(i, int(h)): v for h, v in r.items()})
    rows = []
    for i in inits():
        for L in range(1, 6):
            t = np.array([got[("2m_temperature", i, 24 * L + h)] for h in SAMPLES]) + VARS["2m_temperature"]
            w = np.array([got[("10m_wind_speed", i, 24 * L + h)] for h in SAMPLES]) * 3.6
            r = np.array([rh[(i, 24 * L + h)] for h in SAMPLES])
            for k, code in enumerate(SITES):
                rows.append({"date": i + pd.Timedelta(days=L), "tmax": t[:, k].max(), "tmin": t[:, k].min(), "rh": r[:, k].mean(),
                             "wind": w[:, k].max(), "station": code, "model": "ecmwf_hres", "lead": L})
    pd.DataFrame(rows).to_csv(OUT, index=False, float_format="%.2f")
    print(f"wrote {OUT}: {len(rows)} rows")


if __name__ == "__main__":
    main()
