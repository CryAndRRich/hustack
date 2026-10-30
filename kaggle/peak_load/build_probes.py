import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from preprocess import CLIP, DAY_HOURS, load

EXTRA = Path(__file__).resolve().parent.parent / "extra"
STRUCT = ["level", "hol", "we_day", "we_night", "hum", "t", "t2", "hcos", "hsin"]
HOURS = [f"hr{h}" for h in range(24) if h != 11]
CURV = [f"t2hr{h}" for h in range(24) if h not in (10, 11)]
FAMILIES = ["lgbm", "cat", "xgb", "knn", "mlp", "spline"]
ROUNDS = {
    "v1.2": {"base": "v1.1", "base_lb": 3.261000, "sign": 1, "probe": STRUCT, "solved": []},
    "v1.3": {"base": "v1.2", "base_lb": 2.810473, "sign": -1, "probe": HOURS, "solved": STRUCT[1:7]},
    "v1.4": {"base": "v1.3", "base_lb": 2.654477, "sign": -1, "probe": CURV, "solved": STRUCT[1:7] + HOURS},
    "v2": {"base": "v1.4/blend", "base_lb": 2.503010, "probe": FAMILIES, "solved": []},
}


def components(te):
    t = te.temperature_c.values - 20
    we = te.day_of_week.values >= 5
    day = te.hour.isin(DAY_HOURS).values
    h = 2 * np.pi * te.hour.values / 24
    g = {
        "level": np.ones(len(te)),
        "hol": te.is_holiday.values.astype(float),
        "we_day": (we & day).astype(float),
        "we_night": (we & ~day).astype(float),
        "hum": te.humidity_pct.values - 63.7,
        "t": t,
        "t2": t**2 - np.mean(t**2),
        "hcos": np.cos(h),
        "hsin": np.sin(h),
    }
    for k in range(24):
        hk = te.hour.values == k
        g[f"hr{k}"] = hk.astype(float)
        g[f"t2hr{k}"] = np.where(hk, t**2 - np.mean(t[hk] ** 2), 0.0)
    return g


def base_of(cfg):
    return pd.read_csv(EXTRA / cfg["base"] / "submission.csv").load_mw.values


def main(rnd):
    cfg = ROUNDS[rnd]
    _, te = load()
    b = base_of(cfg)
    free = b < CLIP
    out = EXTRA / rnd
    out.mkdir(parents=True, exist_ok=True)
    g = components(te)
    meta = {}
    for k in cfg["probe"]:
        gk = np.where(free, g[k], 0.0)
        step = cfg["sign"] / np.sqrt(np.mean(gk**2))
        p = np.minimum(b + step * gk, CLIP)
        pd.DataFrame({"id": te.id, "load_mw": p}).to_csv(out / f"probe_{k}.csv", index=False)
        meta[k] = {"step": float(step)}
        print(f"{rnd}/probe_{k}.csv: step {step:.4f}, rms shift {np.sqrt(np.mean((p - b) ** 2)):.3f}")
    (out / "probes.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
