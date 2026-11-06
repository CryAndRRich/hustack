import json
import ssl
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import certifi
import eccodes
import numpy as np
import pandas as pd

from build_nwp import SITES, window_starts

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "external" / "eod_points"
OUT = ROOT / "data" / "external" / "eod_daily.csv"
URL = "https://ecmwf-forecasts.s3.eu-central-1.amazonaws.com/{d:%Y%m%d}/00z/0p4-beta/oper/{d:%Y%m%d}000000-{s}h-oper-fc"
FIRST, LAST = pd.Timestamp("2023-01-18"), pd.Timestamp("2024-02-12")
RH_LEVEL = {"S0": 850, "S6": 850, "S8": 700}
FIELDS = [("2t", None), ("10u", None), ("10v", None)] + [("r", lev) for lev in (1000, 850, 700)]
STEPS = [24 * h + k for h in range(1, 6) for k in range(-3, 19, 3)]
CTX = ssl.create_default_context(cafile=certifi.where())
WORKERS = 6


def get(url, rng=None):
    req = urllib.request.Request(url, headers={"Range": f"bytes={rng[0]}-{rng[0] + rng[1] - 1}"} if rng else {})
    for attempt in range(8):
        try:
            with urllib.request.urlopen(req, timeout=120, context=CTX) as r:
                return r.read()
        except Exception as e:
            print(f"  retry {url} {rng}: {e}")
            time.sleep(2 ** attempt)
    raise RuntimeError(url)


def site_values(msg):
    gid = eccodes.codes_new_from_message(msg)
    try:
        out = []
        for lat, lon in SITES.values():
            near = eccodes.codes_grib_find_nearest(gid, lat, lon, npoints=4)
            w = np.array([1 / max(p["distance"], 1e-3) for p in near])
            out.append(float(np.dot(w, [p["value"] for p in near]) / w.sum()))
        return out
    finally:
        eccodes.codes_release(gid)


def step_points(init, step):
    path = RAW / f"{init:%Y%m%d}_{step:03d}.json"
    if path.exists():
        return json.loads(path.read_text())
    url = URL.format(d=init, s=step)
    index = [json.loads(line) for line in get(url + ".index").decode().splitlines()]
    out = {}
    for param, lev in FIELDS:
        rec = next(r for r in index if r["param"] == param and (lev is None or r.get("levelist") == str(lev)))
        out[f"{param}{lev or ''}"] = site_values(get(url + ".grib2", (rec["_offset"], rec["_length"])))
    path.write_text(json.dumps(out))
    return out


def inits():
    out = []
    for s in window_starts():
        day0 = s + pd.Timedelta(days=9)
        out += [d for d in (day0, day0 - pd.Timedelta(days=5)) if FIRST <= d <= LAST]
    return sorted(set(out))


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    jobs = [(i, s) for i in inits() for s in STEPS]
    print(f"{len(inits())} inits, {len(jobs)} step files")
    with ThreadPoolExecutor(WORKERS) as ex:
        got = dict(zip(jobs, ex.map(lambda j: step_points(*j), jobs)))
    rows = []
    for i in inits():
        for h in range(1, 6):
            steps = [24 * h + k for k in range(-3, 19, 3)]
            t = np.array([got[(i, s)]["2t"] for s in steps]) - 273.15
            ws = np.hypot([got[(i, s)]["10u"] for s in steps], [got[(i, s)]["10v"] for s in steps]) * 3.6
            for k, code in enumerate(SITES):
                rh = np.mean([got[(i, s)][f"r{RH_LEVEL.get(code, 1000)}"][k] for s in steps])
                rows.append({"date": i + pd.Timedelta(days=h), "tmax": t[:, k].max(), "tmin": t[:, k].min(), "rh": rh,
                             "wind": ws[:, k].max(), "station": code, "model": "ecmwf_open", "lead": h})
    pd.DataFrame(rows).to_csv(OUT, index=False, float_format="%.2f")
    print(f"wrote {OUT}: {len(rows)} rows")


if __name__ == "__main__":
    main()
