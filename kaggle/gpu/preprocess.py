from pathlib import Path

import numpy as np
import pandas as pd

BATCH_OFFSET = np.array([-4.6, 2.4, 7.1, -1.8, 0.5, -6.3, 3.9, 8.2, -3.4, 5.6])
NUM = ["util_gpu", "util_mem", "clock_sm_mhz", "power_w", "fan_pct", "temp_inlet_c"]
EMA_COLS = ["util_gpu", "power_w", "fan_pct", "clock_sm_mhz"]
TAUS = [3, 10, 30, 90]
DRIVERS = ["535.161.08", "550.54.15", "560.28.03"]


def data_dir():
    for d in Path("/kaggle/input").glob("**/train.csv"):
        return d.parent
    try:
        return Path(__file__).resolve().parent.parent / "data" / "raw"
    except NameError:
        return Path.cwd() / "data" / "raw"


def offset(df):
    return BATCH_OFFSET[df["device_id"].str[-1].astype(int).to_numpy()]


def ema(g, col, tau, reverse=False):
    s = g[col]
    f = (lambda x: x[::-1].ewm(alpha=1 / tau, adjust=False).mean()[::-1]) if reverse else \
        (lambda x: x.ewm(alpha=1 / tau, adjust=False).mean())
    return s.transform(f)


def features(df):
    df = df.sort_values(["device_id", "step"], ignore_index=True)
    g = df.groupby("device_id")
    df[NUM] = g[NUM].transform(lambda s: s.ffill().bfill())
    out = {c: df[c] for c in NUM}
    out["step"] = df["step"]
    out["frac"] = df["step"] / g["step"].transform("max")
    out["driver"] = df["driver_version"].map({d: i for i, d in enumerate(DRIVERS)})
    out["p_per_fan"] = df["power_w"] / (df["fan_pct"] + 1)
    for c in EMA_COLS:
        for tau in TAUS:
            out[f"{c}_ema{tau}"] = ema(g, c, tau)
        out[f"{c}_rema10"] = ema(g, c, 10, reverse=True)
    for c in ["power_w", "fan_pct", "temp_inlet_c", "util_gpu"]:
        out[f"{c}_dev_mean"] = g[c].transform("mean")
    X = pd.DataFrame(out).astype("float32")
    X["p_per_fan_ema30"] = X["power_w_ema30"] / (X["fan_pct_ema30"] + 1)
    return df, X


def load():
    d = data_dir()
    tr, Xtr = features(pd.read_csv(d / "train.csv"))
    te, Xte = features(pd.read_csv(d / "test.csv"))
    ytr = (tr["temp_hotspot_c"] - offset(tr)).to_numpy()
    return tr, Xtr, ytr, te, Xte
