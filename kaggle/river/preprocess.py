from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "qobs_mm_per_hour"
FILES = ["trainTrackA.csv", "trainTrackB.csv", "testTrackA.csv", "testTrackB.csv", "ExampleSubmission.csv",
         "ext_hourly.csv", "ext_nwm21.csv"]
TEST_START = pd.Timestamp("2017-01-01")
FORCINGS = ["longwave_radiation", "potential_energy", "potential_evaporation", "pressure", "shortwave_radiation",
            "specific_humidity", "temperature", "total_precipitation", "wind_u", "wind_v"]
P_WINDOWS = [3, 6, 12, 24, 48, 72, 168, 336, 720, 2160, 8760]
PET_WINDOWS = [24, 168, 720, 2160, 8760]
WB_WINDOWS = [720, 2160, 8760]
HALF_LIVES = [6, 24, 168, 720]
P_LAGS = [1, 2, 3, 6]
BUCKETS = [50, 150, 400]
NWM = "nwm21_mm_per_hour"
NWM_SHIFTS = [-6, -3, -1, 1, 3, 6, 12]
NWM_WINDOWS = [24, 168, 720]


def find_data():
    root = Path("/kaggle/input")
    hits = {p.name: p for p in root.rglob("*.csv")} if root.exists() else {}
    try:
        d = Path(__file__).resolve().parent.parent / "data"
    except NameError:
        d = Path.cwd() / "data"
    return {f: hits.get(f, d / f) for f in FILES}


def read_competition(paths):
    comp = []
    for f in FILES[:4]:
        d = pd.read_csv(paths[f], dtype={"basin": str})
        d["basin"] = d.basin.str.zfill(8)
        test = f.startswith("test")
        d["date"] = pd.to_datetime(d.date, format="%m/%d/%Y %H:%M" if test else "%Y-%m-%d %H:%M:%S")
        d["part"] = "test" if test else "train"
        comp.append(d)
        print(f"{f}: {d.shape}, {d.date.min()} -> {d.date.max()}")
    return pd.concat(comp, ignore_index=True)


def load():
    paths = find_data()
    comp = read_competition(paths)
    ext = pd.read_csv(paths["ext_hourly.csv"], dtype={"basin": str}, parse_dates=["date"])
    print(f"ext_hourly.csv: {ext.shape}, {ext.date.min()} -> {ext.date.max()}")
    nwm = pd.read_csv(paths["ext_nwm21.csv"], dtype={"basin": str}, parse_dates=["date"])
    ext = ext.merge(nwm, on=["basin", "date"], how="left")
    assert ext[NWM].notna().all(), "NWM v2.1 missing for some hours"


    static = comp.drop(columns=FORCINGS + [TARGET, "date", "part"]).groupby("basin")
    num = static.max(numeric_only=True) - static.min(numeric_only=True)
    assert (num.to_numpy() < 1e-6).all() and (static.nunique().drop(columns=num.columns) == 1).all().all(), \
        "static attributes differ within a basin"
    static = static.first().reset_index()
    tq = comp.loc[comp.part == "train", ["basin", "date", TARGET]]
    df = ext.merge(tq, on=["basin", "date"], how="left", suffixes=("_ext", ""))
    df[TARGET] = df[TARGET].fillna(df.pop(f"{TARGET}_ext"))
    df = df.merge(static, on="basin").sort_values(["basin", "date"]).reset_index(drop=True)
    df["part"] = np.where(df.date >= TEST_START, "test", "train")

    span = df.groupby("basin").date.agg(["min", "max", "size"])
    assert (span["size"] == (span["max"] - span["min"]) / pd.Timedelta("1h") + 1).all(), "series not contiguous"
    te = comp.loc[comp.part == "test", ["basin", "date"]]
    assert len(te.merge(df[["basin", "date"]])) == len(te), "competition test hours missing from the series"
    assert len(tq.merge(df[["basin", "date"]])) == len(tq), "competition train hours missing from the series"
    assert df.loc[df.part == "test", TARGET].isna().all()
    print(f"series: {df.shape}; observed target hours per basin: {df.groupby('basin')[TARGET].count().to_dict()}")
    return df, pd.read_csv(paths["ExampleSubmission.csv"])


def bucket(p, pet, cap):
    s, out_s, out_q = cap / 2, np.empty(len(p)), np.empty(len(p))
    for i in range(len(p)):
        s = max(s + p[i] - pet[i] * s / cap, 0.0)
        q = max(s - cap, 0.0)
        s -= q
        out_s[i], out_q[i] = s, q
    return out_s, out_q


def features(df):
    static = [c for c in df.columns if c not in FORCINGS + [TARGET, NWM, "basin", "date", "part"] and df[c].dtype != object]
    X = df[FORCINGS + static + [NWM]].copy()
    doy = df.date.dt.dayofyear
    X["doy_sin"], X["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)

    g = df.groupby("basin", sort=False)
    p, pet = g.total_precipitation, g.potential_evaporation
    for w in P_WINDOWS:
        X[f"p_{w}h"] = p.transform(lambda s: s.rolling(w, min_periods=1).mean())
    for w in PET_WINDOWS:
        X[f"pet_{w}h"] = pet.transform(lambda s: s.rolling(w, min_periods=1).mean())
    for w in WB_WINDOWS:
        X[f"wb_{w}h"] = X[f"p_{w}h"] - X[f"pet_{w}h"]
    for h in HALF_LIVES:
        X[f"api_{h}h"] = p.transform(lambda s: s.ewm(halflife=h, adjust=False).mean())
    for k in P_LAGS:
        X[f"p_lag{k}"] = p.shift(k)
    nwm = g[NWM]
    for k in NWM_SHIFTS:
        X[f"nwm_shift{k}"] = nwm.shift(k)
    for w in NWM_WINDOWS:
        X[f"nwm_{w}h"] = nwm.transform(lambda s: s.rolling(w, min_periods=1).mean())

    rain, evap = df.total_precipitation.values, np.clip(df.potential_evaporation.values, 0, None)
    for cap in BUCKETS:
        S, Q = np.empty(len(df)), np.empty(len(df))
        for idx in g.indices.values():
            S[idx], Q[idx] = bucket(rain[idx], evap[idx], cap)
        X[f"b{cap}_fill"] = S / cap
        X[f"b{cap}_spill"] = Q
        X[f"b{cap}_spill24"] = pd.Series(Q).groupby(df.basin).transform(
            lambda s: s.ewm(halflife=24, adjust=False).mean()).values
    return X
