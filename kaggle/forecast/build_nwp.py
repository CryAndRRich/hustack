import json
import ssl
import time
import urllib.request
from pathlib import Path

import certifi
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "external" / "openmeteo_prev"
OUT = ROOT / "data" / "external" / "nwp_daily.csv"
URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
SITES = {"S0": (34.08, 74.80), "S1": (21.15, 79.09), "S2": (23.21, 72.68), "S3": (22.57, 88.36), "S4": (26.91, 70.91),
         "S5": (13.08, 80.27), "S6": (25.57, 91.88), "S7": (28.61, 77.21), "S8": (34.15, 77.58), "S9": (8.52, 76.94)}
MODELS = {"jma_gsm": "2018-01-01", "gfs_global": "2021-03-01", "ecmwf_ifs025": "2024-01-01", "icon_global": "2024-01-01",
          "gem_global": "2024-01-01"}
HOURLY = {"temperature_2m": "t", "relative_humidity_2m": "rh", "wind_speed_10m": "ws"}
LEADS = range(1, 6)
CTX = ssl.create_default_context(cafile=certifi.where())
TEST_START, STEP, SPAN = pd.Timestamp("2015-01-01"), 45, 15


def window_starts():
    test = pd.read_csv(ROOT / "data" / "test.csv")
    k = test.window_id.nunique() // len(SITES)
    return TEST_START + pd.to_timedelta(np.arange(k) * STEP, "D")


def fetch(model, start):
    path = RAW / f"{model}_{start:%Y%m%d}.json"
    if path.exists():
        return json.loads(path.read_text())
    lat, lon = zip(*SITES.values())
    q = {"latitude": ",".join(map(str, lat)), "longitude": ",".join(map(str, lon)), "models": model,
         "hourly": ",".join(f"{v}_previous_day{n}" for v in HOURLY for n in LEADS), "timezone": "Asia/Kolkata",
         "start_date": f"{start:%Y-%m-%d}", "end_date": f"{start + pd.Timedelta(days=SPAN - 1):%Y-%m-%d}"}
    url = URL + "?" + "&".join(f"{k}={v}" for k, v in q.items())
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=60, context=CTX) as r:
                data = json.loads(r.read())
            break
        except Exception as e:
            print(f"  retry {model} {start:%Y-%m-%d}: {e}")
            time.sleep(10 * (attempt + 1))
    else:
        raise RuntimeError(f"failed {model} {start}")
    path.write_text(json.dumps(data))
    time.sleep(0.5)
    return data


def daily(model, data):
    rows = []
    for code, loc in zip(SITES, data):
        h = pd.DataFrame(loc["hourly"])
        h["date"] = pd.to_datetime(h.time).dt.normalize()
        for n in LEADS:
            g = h.groupby("date")
            d = pd.DataFrame({"tmax": g[f"temperature_2m_previous_day{n}"].max(),
                              "tmin": g[f"temperature_2m_previous_day{n}"].min(),
                              "rh": g[f"relative_humidity_2m_previous_day{n}"].mean(),
                              "wind": g[f"wind_speed_10m_previous_day{n}"].max()})
            full = g[f"temperature_2m_previous_day{n}"].count() == 24
            d.loc[~full, ["tmax", "tmin"]] = np.nan
            d.loc[g[f"relative_humidity_2m_previous_day{n}"].count() < 24, "rh"] = np.nan
            d.loc[g[f"wind_speed_10m_previous_day{n}"].count() < 24, "wind"] = np.nan
            rows.append(d.reset_index().assign(station=code, model=model, lead=n))
    return pd.concat(rows)


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    out = []
    for model, first in MODELS.items():
        starts = [s for s in window_starts() if s + pd.Timedelta(days=SPAN - 1) >= pd.Timestamp(first)]
        print(f"{model}: {len(starts)} windows")
        for s in starts:
            out.append(daily(model, fetch(model, s)))
    df = pd.concat(out).dropna(subset=["tmax", "tmin", "rh", "wind"], how="all")
    df.to_csv(OUT, index=False, float_format="%.2f")
    print(f"wrote {OUT}: {len(df)} rows")
    print(df.groupby(["model", "lead"])[["tmax", "rh", "wind"]].count().unstack("lead").to_string())


if __name__ == "__main__":
    main()
