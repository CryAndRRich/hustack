from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

WEATHER = ["clear", "mist_cloudy", "light_rain_snow"]
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
FEATS = ["year", "month", "weekday", "is_holiday", "is_workingday", "weather", "temp_c", "feels_like_c",
         "humidity_pct", "wind_speed"]
SENTINEL = -999.0
FEELS_GAP_MAX = 15.0
TEMP_KNOTS = (0, 5, 10, 20, 25, 28)
HUM_KNOTS = (40, 55, 70, 80, 90)
WET_KNOTS = (80, 85, 90)
DAY0 = pd.Timestamp("2011-01-01")


def find_data():
    kaggle = Path("/kaggle/input")
    if kaggle.exists():
        hits = sorted(kaggle.rglob("train.csv"))
        hits = [h for h in hits if (h.parent / "test.csv").exists() and (h.parent / "sample_submission.csv").exists()]
        if hits:
            return hits[0].parent
    try:
        base = Path(__file__).resolve().parent.parent / "data"
    except NameError:
        base = Path.cwd() / "data"
    hits = [h for h in sorted(base.rglob("train.csv")) if (h.parent / "test.csv").exists()]
    if not hits:
        raise FileNotFoundError("train.csv / test.csv not found under /kaggle/input or data/")
    return hits[0].parent


def _fill_by_regression(df, target, cols, log):
    miss = df[target].isna()
    if not miss.any():
        return
    ok = df[[target, *cols]].notna().all(1)
    m = LinearRegression().fit(df.loc[ok, cols], df.loc[ok, target])
    assert df.loc[miss, cols].notna().all().all(), f"{target}: predictors missing too"
    df.loc[miss, target] = m.predict(df.loc[miss, cols])
    log(f"  {target}: {int(miss.sum())} filled by OLS on {cols} (in-sample R2 {m.score(df.loc[ok, cols], df.loc[ok, target]):.3f})")


def load_clean(log):
    root = find_data()
    tr = pd.read_csv(root / "train.csv")
    te = pd.read_csv(root / "test.csv")
    sample = pd.read_csv(root / "sample_submission.csv")
    log(f"data {root}: train {tr.shape}, test {te.shape}, sample {sample.shape}")
    assert list(te.columns) == ["row_id", "date", *FEATS], "unexpected test columns"
    assert list(tr.columns) == ["row_id", "date", *FEATS, "casual", "registered", "rentals"], "unexpected train columns"
    assert set(sample["row_id"]) == set(te["row_id"]) and len(sample) == len(te), "sample ids differ from test ids"
    report = {}

    n0 = len(tr)
    tr = tr.drop_duplicates().reset_index(drop=True)
    report["train_duplicates_dropped"] = n0 - len(tr)
    assert not tr["row_id"].duplicated().any() and not tr["date"].duplicated().any(), "conflicting duplicates remain"
    bad = tr["rentals"] != tr["casual"] + tr["registered"]
    report["rentals_fixed"] = tr.loc[bad, ["date", "rentals"]].assign(fixed=tr.loc[bad, "casual"] + tr.loc[bad, "registered"]).to_dict("records")
    tr.loc[bad, "rentals"] = tr.loc[bad, "casual"] + tr.loc[bad, "registered"]

    tr["part"], te["part"] = "train", "test"
    df = pd.concat([tr, te], ignore_index=True)
    w = df["weather"].astype(str).str.strip().str.lower().replace({"cleer": "clear"})
    report["weather_fixed"] = int((w != df["weather"]).sum())
    df["weather"] = w
    assert set(df["weather"]) <= set(WEATHER), f"unknown weather {set(df['weather']) - set(WEATHER)}"
    dt = pd.to_datetime(df["date"], format="%Y-%m-%d")
    assert (dt.dt.year == df["year"]).all() and (dt.dt.month == df["month"]).all()
    assert (dt.dt.strftime("%a") == df["weekday"]).all(), "weekday disagrees with date"
    assert dt.is_unique, "a date appears twice across train and test"
    df["t"] = (dt - DAY0).dt.days

    for c in ("temp_c", "wind_speed", "humidity_pct", "feels_like_c"):
        s = df[c] == SENTINEL
        report[f"{c}_sentinel"] = int(s.sum())
        df.loc[s, c] = np.nan
    z = df["humidity_pct"] == 0
    report["humidity_zero"] = int(z.sum())
    df.loc[z, "humidity_pct"] = np.nan
    gap = (df["feels_like_c"] - df["temp_c"]).abs() > FEELS_GAP_MAX
    report["feels_like_inconsistent"] = df.loc[gap, ["date", "temp_c", "feels_like_c"]].to_dict("records")
    df.loc[gap, "feels_like_c"] = np.nan
    report["missing_before_impute"] = df[["temp_c", "feels_like_c", "humidity_pct", "wind_speed"]].isna().sum().to_dict()
    log(f"cleaning: {report}")

    df = df.sort_values("t").reset_index(drop=True)
    for c in ("humidity_pct", "wind_speed"):
        n = int(df[c].isna().sum())
        df[c] = df[c].interpolate(method="linear", limit_direction="both")
        log(f"  {c}: {n} filled by linear interpolation over the date (all 731 days present)")
    _fill_by_regression(df, "temp_c", ["feels_like_c", "humidity_pct", "wind_speed"], log)
    _fill_by_regression(df, "feels_like_c", ["temp_c", "humidity_pct", "wind_speed"], log)
    assert df[["temp_c", "feels_like_c", "humidity_pct", "wind_speed"]].notna().all().all()
    for c, lo, hi in (("temp_c", -20, 45), ("feels_like_c", -25, 50), ("humidity_pct", 1, 100), ("wind_speed", 0, 60)):
        assert df[c].between(lo, hi).all(), f"{c} out of range after cleaning"
    train = df[df["part"] == "train"].reset_index(drop=True)
    test = df[df["part"] == "test"].reset_index(drop=True)
    assert len(test) == len(sample) and train["rentals"].notna().all()
    return train, test, sample, report


def _nth_weekday(year, month, weekday, n):
    days = pd.date_range(f"{year}-{month:02d}-01", periods=pd.Period(f"{year}-{month:02d}").days_in_month)
    hits = days[days.dayofweek == weekday]
    return hits[n]


def holiday_groups(df):
    dt = pd.to_datetime(df["date"])
    groups = {"thanksgiving": [], "july4": [], "winter_monday": [], "summer_monday": []}
    for y in (2011, 2012):
        tg = _nth_weekday(y, 11, 3, 3)
        groups["thanksgiving"] += [tg, tg + pd.Timedelta(days=1)]
        groups["july4"].append(pd.Timestamp(f"{y}-07-04"))
        groups["winter_monday"] += [_nth_weekday(y, 1, 0, 2), _nth_weekday(y, 2, 0, 2)]
        groups["summer_monday"] += [_nth_weekday(y, 5, 0, -1), _nth_weekday(y, 9, 0, 0)]
    G = pd.DataFrame(index=df.index)
    for g, days in groups.items():
        G[f"hol_{g}"] = dt.isin(days).astype(float)
    G["hol_xmas"] = (((dt.dt.month == 12) & (dt.dt.day >= 23)) | ((dt.dt.month == 1) & (dt.dt.day == 1))).astype(float)
    special = G.max(axis=1) > 0
    G["hol_other"] = ((df["is_holiday"] == 1) & ~special).astype(float)
    hol_days = set(dt[df["is_holiday"] == 1]) | {d for v in groups.values() for d in v}
    near = dt.apply(lambda d: (d - pd.Timedelta(days=1)) in hol_days or (d + pd.Timedelta(days=1)) in hol_days)
    G["hol_near"] = (near & ~special & (df["is_holiday"] == 0)).astype(float)
    return G


def features(df, knot=None, inter=False, calendar=True, hinge=False, scale=False, v3=False):
    X = pd.DataFrame(index=df.index)
    y2012 = (df["year"] == 2012).astype(float)
    if knot is None or calendar:
        X["y2012"] = y2012
        for m in range(2, 13):
            X[f"m{m}"] = (df["month"] == m).astype(float)
    if knot is not None:
        X["t"] = df["t"] / 365.0
        for k in np.arange(knot, 731, knot):
            X[f"h{k}"] = np.maximum(0.0, df["t"] - k) / 365.0
    for d in WEEKDAYS[1:]:
        X[f"wd_{d}"] = (df["weekday"] == d).astype(float)
    if v3:
        X = pd.concat([X, holiday_groups(df)], axis=1)
        light = (df["weather"] == "light_rain_snow").astype(float)
        X["w_light"], X["w_light_y12"] = light, light * y2012
    else:
        X["holiday"] = df["is_holiday"].astype(float)
    X["working"] = df["is_workingday"].astype(float)
    W = pd.DataFrame(index=df.index)
    for w in (WEATHER[1:2] if v3 else WEATHER[1:]):
        W[f"w_{w}"] = (df["weather"] == w).astype(float)
    tc = df["temp_c"] - 15.0
    hum = df["humidity_pct"] - 60.0
    if hinge:
        W["temp"] = tc
        for k in TEMP_KNOTS:
            W[f"temp_h{k}"] = np.maximum(0.0, df["temp_c"] - k)
        W["feels"] = df["feels_like_c"] - df["temp_c"]
        W["hum"] = hum
        for k in HUM_KNOTS:
            W[f"hum_h{k}"] = np.maximum(0.0, df["humidity_pct"] - k)
        for w in (WEATHER[1:2] if v3 else WEATHER[1:]):
            W[f"hum_x_{w}"] = hum * W[f"w_{w}"]
    else:
        W["temp"], W["temp2"] = tc, tc ** 2
        W["hum"] = hum
    W["wind"] = df["wind_speed"] - 12.0
    if v3:
        mist = (df["weather"] == "mist_cloudy").astype(float)
        for k in WET_KNOTS:
            W[f"wet_h{k}"] = mist * np.maximum(0.0, df["humidity_pct"] - k)
        W["wet_x_wind"] = W[f"wet_h{WET_KNOTS[0]}"] * W["wind"]
    blocks = [X, W]
    if inter:
        blocks.append(W.mul(X["working"], axis=0).add_prefix("work_x_"))
    if scale:
        blocks.append(pd.concat([W, X[["working"]]], axis=1).mul(y2012, axis=0).add_prefix("y12_x_"))
    return pd.concat(blocks, axis=1)
