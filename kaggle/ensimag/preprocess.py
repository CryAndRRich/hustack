import importlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OK_LIMIT = 1000.0
TZ = "Europe/Berlin"
SERIES = ["wind", "solar", "load", "resid"]
STEPS = (1, 2, 4)

STATE_POP_M = {"BW": 11.3, "BY": 13.4, "BE": 3.8, "BB": 2.6, "HB": 0.7, "HH": 1.9, "HE": 6.4, "MV": 1.6,
               "NI": 8.1, "NW": 18.1, "RP": 4.2, "SL": 1.0, "SN": 4.1, "ST": 2.2, "SH": 3.0, "TH": 2.1}
EXT_GROUPS = {
    "error": ["smard_onshore_fc", "smard_pv_fc", "onshore_fc_diff", "pv_fc_diff", "ren_fc_diff",
              "onshore_fc_diff_dmean", "pv_fc_diff_dmean", "ren_fc_diff_dmean"],
}
ERROR2_STEPS = {"pv": (1, 2, 4), "on": (1, 2, 4), "ren": (1, 2, 4, 8)}
EXT_GROUPS["error2"] = [f"{k}_fc_diff_n" for k in ERROR2_STEPS] + \
    [f"{k}_fc_diff_n_b{s}" for k, steps in ERROR2_STEPS.items() for s in steps] + \
    [f"{k}_fc_diff_n_{w}" for k in ("pv", "ren") for w in ("f1", "hdev")]

DESIGNED_NAN_SUFFIXES = ("_f1",)


def get_data_dir() -> Path:
    kaggle = Path("/kaggle/input")
    if kaggle.exists():
        for p in sorted(kaggle.rglob("train.csv")):
            with open(p, encoding="utf-8") as f:
                head = f.readline()
            if "spread" in head and "solar" in head and (p.parent / "imbalances.csv").exists():
                return p.parent
        raise FileNotFoundError("no train.csv with a spread column (and imbalances.csv beside it) under /kaggle/input")
    try:
        return Path(__file__).resolve().parent.parent / "data" / "competition"
    except NameError:
        return Path.cwd() / "data" / "competition"


def load_data(data_dir: Path):
    train = pd.read_csv(data_dir / "train.csv", parse_dates=["date"])
    test = pd.read_csv(data_dir / "test.csv", parse_dates=["date"])
    imb = pd.read_csv(data_dir / "imbalances.csv", parse_dates=["date"])
    sample = pd.read_csv(data_dir / "sample.csv")
    assert list(train.columns) == ["date", "wind", "solar", "load", "spread"], list(train.columns)
    assert list(test.columns) == ["ID", "date", "wind", "solar", "load"], list(test.columns)
    assert list(sample.columns) == ["ID", "forecast"], list(sample.columns)
    assert len(imb) == len(train) and (imb["date"].values == train["date"].values).all(), \
        "imbalances.csv is not row-aligned with train.csv"
    assert (sample["ID"].values == test["ID"].values).all(), "sample.csv IDs differ from test.csv IDs"
    for name, d in (("train", train), ("test", test)):
        assert d["date"].is_monotonic_increasing and not d["date"].duplicated().any(), f"{name} dates not strictly increasing"
        assert d[["wind", "solar", "load"]].notna().all().all(), f"NaN in {name} features"
    assert train["date"].max() < test["date"].min(), "test does not follow train in time"
    train["imbalances"] = imb["imbalances"].values
    print(f"data: train {len(train)} rows {train['date'].min()} -> {train['date'].max()}, "
          f"test {len(test)} rows {test['date'].min()} -> {test['date'].max()}")
    return train, test, sample


def pip_import(name: str):

    try:
        mod = importlib.import_module(name)
    except ImportError:
        print(f"{name} not installed — pip install {name}")
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", name], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"pip install {name} failed — turn the notebook's Internet ON\n" + r.stderr[-2000:])
        mod = importlib.import_module(name)
    print(f"{name} {mod.__version__}")
    return mod


def holiday_share(days: pd.DatetimeIndex) -> pd.Series:

    holidays = pip_import("holidays")
    years = range(days.min().year - 1, days.max().year + 2)
    total = sum(STATE_POP_M.values())
    share = pd.Series(0.0, index=days)
    for state, pop in STATE_POP_M.items():
        hol = holidays.country_holidays("DE", subdiv=state, years=years)
        share += np.array([d.date() in hol for d in days], dtype=float) * pop / total
    return share.round(4)


def day_calendar(days: pd.DatetimeIndex) -> pd.DataFrame:
    full = pd.date_range(days.min() - pd.Timedelta(days=1), days.max() + pd.Timedelta(days=1), freq="D")
    share = holiday_share(full)
    off = (share >= 0.999) | (full.dayofweek >= 5)
    cal = pd.DataFrame(index=full)
    cal["hol_share"] = share.values
    cal["bridge"] = (~off & off.shift(1, fill_value=False) & off.shift(-1, fill_value=False)).astype(float)
    md = full.month * 100 + full.day
    cal["xmas"] = ((md >= 1224) | (md <= 101)).astype(float)
    n_nat = int((share >= 0.999).sum())
    print(f"calendar: {n_nat} nationwide holidays, {int(((share > 0) & (share < 0.999)).sum())} regional, "
          f"{int(cal['bridge'].sum())} bridge days over {len(full)} days")
    return cal.reindex(days)


def build_features(train: pd.DataFrame, test: pd.DataFrame, ext: pd.DataFrame):
\
\
\
\

    both = pd.concat([train[["date", "wind", "solar", "load"]], test[["date", "wind", "solar", "load"]]],
                     ignore_index=True)
    grid = pd.date_range(both["date"].min(), both["date"].max(), freq="15min")
    g = both.set_index("date").reindex(grid)
    g.index.name = "date"
    print(f"grid: {len(grid)} slots, {int(g['load'].isna().sum())} missing")
    g["resid"] = g["load"] - g["wind"] - g["solar"]

    loc = g.index.tz_localize("UTC").tz_convert(TZ)
    ldate = pd.DatetimeIndex(loc.tz_localize(None).floor("D"))
    g["qh"] = g.index.minute // 15
    g["lhour"] = loc.hour
    g["dow"] = loc.dayofweek
    g["month"] = loc.month
    cal = day_calendar(pd.date_range(ldate.min(), ldate.max(), freq="D"))
    for c in cal.columns:
        g[c] = cal[c].reindex(ldate).values

    ld = pd.Series(ldate, index=g.index)
    hkey = pd.Series(g.index.floor("h"), index=g.index)
    hours = pd.DatetimeIndex(sorted(hkey.unique()))
    h_ld = pd.Series(pd.DatetimeIndex(hours.tz_localize("UTC").tz_convert(TZ).tz_localize(None).floor("D")), index=hours)
    next_same_day = (h_ld.shift(-1) == h_ld).values

    for s in SERIES:
        x = g[s]
        for k in STEPS:
            g[f"{s}_b{k}"] = x - x.shift(k)
            g[f"{s}_f{k}"] = (x.shift(-k) - x).where(ld.shift(-k) == ld)
        g[f"{s}_hdev"] = x - x.groupby(hkey).transform("mean")
        hm = x.groupby(hkey).mean().reindex(hours)
        prev = hm.shift(1)
        nxt = hm.shift(-1).where(next_same_day)
        g[f"{s}_hup"] = (hm - prev).reindex(hkey).values
        g[f"{s}_hnext"] = (nxt - hm).reindex(hkey).values
        g[f"{s}_ramp"] = (nxt - prev).reindex(hkey).values
        g[f"{s}_ddev"] = x - x.groupby(ld).transform("mean")

    for s in ("resid", "load"):
        g[f"{s}_dmean"] = g[s].groupby(ld).transform("mean")
        g[f"{s}_dmax"] = g[s].groupby(ld).transform("max")
        g[f"{s}_dmin"] = g[s].groupby(ld).transform("min")

    for s in ("solar", "wind"):
        dmax = g[s].groupby(ld).max()
        cap = dmax.rolling(30, min_periods=1).max().replace(0, np.nan)
        capv = cap.reindex(ldate).values
        g[f"{s}_n"] = g[s] / capv
        g[f"{s}_b1n"] = g[f"{s}_b1"] / capv
        g[f"{s}_f1n"] = g[f"{s}_f1"] / capv
        g[f"{s}_dmean_n"] = g[s].groupby(ld).transform("mean") / capv
    g["ren_share"] = (g["wind"] + g["solar"]) / g["load"]
    g["resid_share"] = g["resid"] / g["load"]

    groups = {"v1": list(g.columns), **EXT_GROUPS}
    g = pd.concat([g, add_external(g, ext, ldate, ld)], axis=1)
    features = list(dict.fromkeys(f for fs in groups.values() for f in fs))
    missing = [f for f in features if f not in g.columns]
    assert not missing, f"feature groups name columns that were not built: {missing}"
    x_tr = g.loc[pd.DatetimeIndex(train["date"]), features].reset_index(drop=True)
    x_te = g.loc[pd.DatetimeIndex(test["date"]), features].reset_index(drop=True)
    assert len(x_tr) == len(train) and len(x_te) == len(test)
    ext_cols = [f for f in features if f not in groups["v1"] and not f.endswith(DESIGNED_NAN_SUFFIXES)]
    nan_tr, nan_te = x_tr[ext_cols].isna().mean(), x_te[ext_cols].isna().mean()
    bad = sorted(set(nan_tr[nan_tr > 0.01].index) | set(nan_te[nan_te > 0.01].index))
    assert not bad, f"external columns > 1% NaN on train/test rows: {bad} — stale dataset, re-upload data.zip"
    print(f"external: {len(ext_cols)} columns, max NaN share train {nan_tr.max():.4f}, test {nan_te.max():.4f}")
    meta = pd.DataFrame({"date": train["date"].values,
                         "year": train["date"].dt.year.values,
                         "month": train["date"].dt.month.values,
                         "ldate": ld.loc[pd.DatetimeIndex(train["date"])].values})
    print(f"features: {len(features)} ({', '.join(f'{k} {len(v)}' for k, v in groups.items())}); "
          f"NaN share train {x_tr.isna().mean().mean():.4f}, test {x_te.isna().mean().mean():.4f}")
    return x_tr, x_te, meta, groups


def add_external(g: pd.DataFrame, ext: pd.DataFrame, ldate: pd.DatetimeIndex, ld: pd.Series) -> pd.DataFrame:
\

    q = ext.reindex(g.index)
    e = pd.DataFrame(index=g.index)
    e["smard_onshore_fc"] = q["smard_wind_fc"] - q["smard_offshore_fc"]
    e["smard_pv_fc"] = q["smard_pv_fc"]
    e["onshore_fc_diff"] = e["smard_onshore_fc"] - g["wind"]
    e["pv_fc_diff"] = e["smard_pv_fc"] - g["solar"]
    e["ren_fc_diff"] = e["onshore_fc_diff"] + e["pv_fc_diff"]
    for c in ("onshore_fc_diff", "pv_fc_diff", "ren_fc_diff"):
        e[f"{c}_dmean"] = e[c].groupby(ld).transform("mean")


    cap = {s: g[s].groupby(ld).max().rolling(30, min_periods=1).max().replace(0, np.nan).reindex(ldate).values
           for s in ("solar", "wind")}
    hkey = pd.Series(g.index.floor("h"), index=g.index)
    n = {"pv": e["pv_fc_diff"] / cap["solar"], "on": e["onshore_fc_diff"] / cap["wind"]}
    n["ren"] = e["ren_fc_diff"] / (cap["solar"] + cap["wind"])
    e2 = {}
    for k, x in n.items():
        e2[f"{k}_fc_diff_n"] = x
        for s in ERROR2_STEPS[k]:
            e2[f"{k}_fc_diff_n_b{s}"] = x - x.shift(s)
        if k in ("pv", "ren"):
            e2[f"{k}_fc_diff_n_f1"] = (x.shift(-1) - x).where(ld.shift(-1) == ld)
            e2[f"{k}_fc_diff_n_hdev"] = x - x.groupby(hkey).transform("mean")
    return pd.concat([e, pd.DataFrame(e2, index=g.index)], axis=1)


def get_ext_dir() -> Path:
    kaggle = Path("/kaggle/input")
    if kaggle.exists():
        for p in sorted(kaggle.rglob("ext_smard_qh.csv")):
            with open(p, encoding="utf-8") as f:
                if "smard_pv_fc" in f.readline():
                    return p.parent
        raise FileNotFoundError("no ext_smard_qh.csv under /kaggle/input — attach the dataset made from data/data.zip")
    try:
        return Path(__file__).resolve().parent.parent / "data"
    except NameError:
        return Path.cwd() / "data"


def load_external(ext_dir: Path) -> pd.DataFrame:
    ext = pd.read_csv(ext_dir / "ext_smard_qh.csv", parse_dates=["date"]).set_index("date")
    miss = [c for c in ("smard_wind_fc", "smard_offshore_fc", "smard_pv_fc") if c not in ext.columns]
    assert not miss, f"ext_smard_qh.csv lacks {miss} — stale dataset, re-upload data.zip"
    print(f"external data from {ext_dir}: {ext.shape} {ext.index.min()} -> {ext.index.max()}")
    return ext
