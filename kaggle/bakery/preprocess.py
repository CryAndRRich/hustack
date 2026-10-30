from pathlib import Path

import numpy as np
import pandas as pd

RIVER_X = 6.0
EVENT_TYPES = ["market", "concert", "sport"]
EVENT_BINS = [0, 0.5, 1, 1.5, 2, 3]
GAP_START = "2025-07-21"


def data_dir():
    root = Path("/kaggle/input")
    for p in sorted(root.rglob("train.csv")) if root.exists() else []:
        if "units_sold" in pd.read_csv(p, nrows=0).columns:
            return p.parent
    try:
        return Path(__file__).resolve().parent.parent / "data"
    except NameError:
        return Path.cwd() / "data"


def attendance(text):
    num = pd.Series(text).str.extract(r"(\d[\d .]*\d|\d)")[0].str.strip()
    val = num.str.replace(" ", "").astype(float)
    return np.where(num.str.contains(r"\."), val * 1000, val)


def walk(a, b, bridges):
    direct = np.hypot(a[:, None, 0] - b[None, :, 0], a[:, None, 1] - b[None, :, 1])
    via = np.min([np.hypot(*(a - q).T)[:, None] + np.hypot(*(b - q).T)[None, :] for q in bridges], axis=0)
    same = (a[:, None, 0] < RIVER_X) == (b[None, :, 0] < RIVER_X)
    return np.where(same, direct, via)


def load():
    d = data_dir()
    t = {k: pd.read_csv(d / f"{k}.csv") for k in
         ["train", "test", "shops", "products", "weather", "events", "bridges", "sample_submission"]}
    print(f"data {d}: train {t['train'].shape}, test {t['test'].shape}")
    return t


def build(t):
    tr, te = t["train"].copy(), t["test"].copy()
    last = tr.date.max()
    tr["split"] = "train"
    te["split"] = np.where(te.date > last, "future", "gap")
    df = pd.concat([tr, te], ignore_index=True)
    n = len(df)
    df = df.merge(t["shops"], on="shop_id", how="left").merge(t["products"][["product_id", "regular_price"]],
                                                              on="product_id", how="left")
    df = df.merge(t["weather"], on="date", how="left")
    assert len(df) == n and df.temperature_forecast_c.notna().all()
    dates = np.sort(df.date.unique())
    df["day"] = np.searchsorted(dates, df.date)
    df["dow"] = pd.to_datetime(df.date).dt.dayofweek
    df["shop"] = df.shop_id.astype("category").cat.codes.astype(int)
    df["prod"] = df.product_id.astype("category").cat.codes.astype(int)
    df["disc"] = np.log(df.price / df.regular_price)
    doy = pd.to_datetime(df.date).dt.dayofyear.values
    df["season_sin"], df["season_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    assert df.groupby(["date", "shop_id", "product_id"]).size().max() == 1

    ev = t["events"].copy()
    ev["att"] = attendance(ev.announcement) / 1000
    sh = t["shops"].set_index("shop_id")
    dist = walk(sh[["x_km", "y_km"]].values, ev[["x_km", "y_km"]].values, t["bridges"][["x_km", "y_km"]].values)
    rows = []
    for j, e in ev.iterrows():
        for i, s in enumerate(sh.index):
            b = np.searchsorted(EVENT_BINS, dist[i, j], side="right") - 1
            if b < len(EVENT_BINS) - 1:
                rows.append((e.date, s, f"ev_{e.event_type}_{b}", e.att))
    names = [f"ev_{k}_{b}" for k in EVENT_TYPES for b in range(len(EVENT_BINS) - 1)]
    near = pd.DataFrame(rows, columns=["date", "shop_id", "feat", "att"])
    near["sqrt_att"] = np.sqrt(near.att)
    out = []
    for col in ("att", "sqrt_att"):
        M = near.pivot_table(index=["date", "shop_id"], columns="feat", values=col, aggfunc="sum")
        M = M.reindex(columns=names).fillna(0.0)
        M = df[["date", "shop_id"]].merge(M, left_on=["date", "shop_id"], right_index=True, how="left")[names]
        assert len(M) == n
        out.append(M.fillna(0.0).values)
    print("rows by split", df.split.value_counts().to_dict(), "| shop-days near an event", (out[0].sum(1) > 0).mean().round(3))
    return df, out[0], out[1], names
