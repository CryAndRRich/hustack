import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from build_eod import get, site_values
from build_nwp import SITES, window_starts

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "external" / "gefs_points"
OUT = ROOT / "data" / "external" / "gefs_daily.csv"
URL = ("https://noaa-gefs-retrospective.s3.amazonaws.com/GEFSv12/reforecast/{d:%Y}/{d:%Y%m%d}00/{m}/Days:1-10/"
       "{f}_{d:%Y%m%d}00_{m}.grib2")
LAST = pd.Timestamp("2015-12-31")
CALIB_YEARS = (1, 2)
MEMBERS = ["c00", "p01", "p02", "p03", "p04"]
FIELDS = {"tmp_2m": "TMP:2 m", "ugrd_hgt": "UGRD:10 m", "vgrd_hgt": "VGRD:10 m", "spfh_2m": "SPFH:2 m", "pres_sfc": "PRES:surface"}
STEPS = list(range(21, 139, 3))
WORKERS = 6


def field_points(url, tag):
    index = [line.split(":") for line in get(url + ".idx").decode().splitlines()]
    offsets = [int(r[1]) for r in index] + [None]
    want = {int(r[5].split()[0]): k for k, r in enumerate(index)
            if f"{r[3]}:{r[4]}".startswith(tag) and r[5].endswith("hour fcst") and int(r[5].split()[0]) in STEPS}
    lo, hi = min(want.values()), max(want.values())
    blob = get(url, (offsets[lo], offsets[hi + 1] - offsets[lo]))
    return {s: site_values(blob[offsets[k] - offsets[lo]:offsets[k + 1] - offsets[lo]]) for s, k in want.items()}


def run_points(init, member):
    path = RAW / f"{init:%Y%m%d}_{member}.json"
    if path.exists():
        return {f: {int(s): v for s, v in d.items()} for f, d in json.loads(path.read_text()).items()}
    out = {f: field_points(URL.format(d=init, m=member, f=f), tag) for f, tag in FIELDS.items()}
    path.write_text(json.dumps(out))
    return out


def daily(p):
    out = np.zeros((5, len(SITES), 4))
    for h in range(1, 6):
        steps = [24 * h + k for k in range(-3, 19, 3)]
        t = np.array([p["tmp_2m"][s] for s in steps]) - 273.15
        ws = np.hypot([p["ugrd_hgt"][s] for s in steps], [p["vgrd_hgt"][s] for s in steps]) * 3.6
        q, pr = np.array([p["spfh_2m"][s] for s in steps]), np.array([p["pres_sfc"][s] for s in steps])
        rh = np.clip(100 * q * pr / (0.622 + 0.378 * q) / (611.2 * np.exp(17.67 * t / (t + 243.5))), 0, 100)
        out[h - 1] = np.stack([t.max(0), t.min(0), rh.mean(0), ws.max(0)], 1)
    return out


def inits():
    out = []
    for s in window_starts():
        day0 = s + pd.Timedelta(days=9)
        out += [d for d in (day0, day0 - pd.Timedelta(days=5)) if d <= LAST]
    return sorted(set(out + [d - pd.DateOffset(years=y) for d in out for y in CALIB_YEARS]))


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    jobs = [(i, m) for i in inits() for m in MEMBERS]
    print(f"{len(inits())} inits, {len(jobs)} runs")
    with ThreadPoolExecutor(WORKERS) as ex:
        got = dict(zip(jobs, ex.map(lambda j: run_points(*j), jobs)))
    rows = []
    for i in inits():
        d = np.mean([daily(got[(i, m)]) for m in MEMBERS], 0)
        for h in range(1, 6):
            for k, code in enumerate(SITES):
                rows.append({"date": i + pd.Timedelta(days=h), **dict(zip(["tmax", "tmin", "rh", "wind"], d[h - 1, k])),
                             "station": code, "model": "gefs", "lead": h})
    pd.DataFrame(rows).to_csv(OUT, index=False, float_format="%.2f")
    print(f"wrote {OUT}: {len(rows)} rows")


if __name__ == "__main__":
    main()
