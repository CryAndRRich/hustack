import json
from pathlib import Path

import numpy as np
import pandas as pd

WIDTH = 56
RAIN_WINDOW = 72
STATIC = ["sm_2in", "sm_20in", "season", "T_air_C", "srad_Wm2"]
LAYERS = ["fc1", "fc2", "fc3", "fc4", "fc5", "fc6", "out"]
CANDIDATES = ["fc2", "fc3", "fc4", "fc5", "fc6"]
N_TRAIN = 15388
N_HELDOUT = 3298
N_EXAMPLES = 21983

HOST_RAIN_MM = {"2026-04-14 04:00": 0.16 * 25.4}

def get_data_dir():
    kaggle = Path("/kaggle/input")
    if kaggle.exists():
        for p in sorted(kaggle.rglob("train.csv")):
            with open(p, encoding="utf-8") as f:
                if "streamflow_mm_hr" in f.readline():
                    return p.parent
        raise FileNotFoundError("no train.csv with streamflow_mm_hr under /kaggle/input")
    try:
        return Path(__file__).resolve().parent.parent / "data"
    except NameError:
        return Path.cwd() / "data"

def find(d, name):
    for p in (d / name, d / "external" / name):
        if p.exists():
            return p
    hits = sorted(Path("/kaggle/input").rglob(name)) if Path("/kaggle/input").exists() else []
    if not hits:
        raise FileNotFoundError(name)
    return hits[0]

def out_dir():
    return Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")

def load_weights(d):
    with np.load(find(d, "streamflow_model_bug.npz")) as z:
        return {k: z[k].astype(np.float64) for k in z.files}

def load_data(d):
    scaler = json.load(open(find(d, "feature_scaler.json")))
    mu, sig = np.array(scaler["mu"]), np.array(scaler["sig"])
    df = pd.read_csv(d / "train.csv", parse_dates=["datetime"]).sort_values("datetime").reset_index(drop=True)
    rain = df["rain_mm"].to_numpy(np.float32)
    windows = np.lib.stride_tricks.sliding_window_view(rain, RAIN_WINDOW)
    dfw = df.iloc[RAIN_WINDOW - 1:].reset_index(drop=True)
    assert dfw[STATIC].notna().all().all() and np.isfinite(windows).all()
    raw = np.concatenate([windows, dfw[STATIC].to_numpy(np.float32)], axis=1).astype(np.float32)
    X = ((raw - mu.astype(np.float32)) / sig.astype(np.float32)).astype(np.float64)
    dfw["rain24"] = windows[:, -24:].sum(1)
    dfw["rain72"] = windows.sum(1)
    dfw["split"] = np.where(np.arange(len(dfw)) < N_TRAIN, "train", "val")
    assert len(dfw) == 18685
    return X, dfw["streamflow_mm_hr"].to_numpy(np.float64), dfw

def load_heldout(d):
    scaler = json.load(open(find(d, "feature_scaler.json")))
    mu, sig = np.array(scaler["mu"]), np.array(scaler["sig"])
    pub = pd.read_csv(d / "train.csv", parse_dates=["datetime"]).sort_values("datetime").drop(columns="streamflow_mm_hr")
    ho = pd.read_csv(find(d, "heldout.csv"), parse_dates=["datetime"])
    assert len(ho) == N_HELDOUT and ho["datetime"].min() > pub["datetime"].max()
    for t, mm in HOST_RAIN_MM.items():
        hit = ho["datetime"] == pd.Timestamp(t)
        assert hit.sum() == 1
        ho.loc[hit, "rain_mm"] = mm
    allr = pd.concat([pub, ho[pub.columns]], ignore_index=True)
    windows = np.lib.stride_tricks.sliding_window_view(allr["rain_mm"].to_numpy(np.float32), RAIN_WINDOW)
    raw = np.concatenate([windows, allr.iloc[RAIN_WINDOW - 1:][STATIC].to_numpy(np.float32)], axis=1).astype(np.float32)
    assert len(raw) == N_EXAMPLES
    gap = np.abs(raw.astype(np.float64).mean(0) - mu) / sig
    assert gap.max() < 0.01, f"rebuilt held-out does not reproduce the scaler: max |mean - mu| / sig = {gap.max():.4f}"
    X = ((raw - mu.astype(np.float32)) / sig.astype(np.float32)).astype(np.float64)
    meta = ho.reset_index(drop=True)
    return X[-N_HELDOUT:], meta, gap

def forward(W, X, edits=None):
    h, acts, pre = X, [X], []
    for name in LAYERS:
        w = W[name + ".weight"] if not edits or name not in edits else edits[name]
        z = h @ w.T + W[name + ".bias"]
        pre.append(z)
        h = z if name == "out" else np.maximum(z, 0.0)
        acts.append(h)
    return h[:, 0], acts, pre

def metrics(pred, y):
    r = pred - y
    return {"RMSE": float(np.sqrt(np.mean(r ** 2))), "MAE": float(np.mean(np.abs(r))), "bias": float(r.mean()),
            "NSE": float(1 - np.sum(r ** 2) / np.sum((y - y.mean()) ** 2))}
