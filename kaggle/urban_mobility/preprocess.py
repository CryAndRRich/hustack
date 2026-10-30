from pathlib import Path

import numpy as np
import pandas as pd

H = 30
START = pd.Timestamp("2021-04-15")
ANOMALIES = [("2021-09-07", "2021-10-01"), ("2021-11-08", "2021-11-11")]
IMPUTE_WEEKS = 4
HOLIDAYS = pd.to_datetime(
    ["2021-05-03", "2021-05-04", "2021-05-05", "2021-05-06", "2021-05-07", "2021-05-10", "2021-06-14",
     "2021-11-01", "2021-11-02", "2021-11-03", "2021-11-04", "2021-11-05", "2021-12-31"]
    + [f"2022-01-{d:02d}" for d in range(1, 10)])
COVARIATES = ["off", "long_off", "last_off", "pre_long"]


def get_data_dir() -> Path:
    kaggle = Path("/kaggle/input")
    if kaggle.exists():
        for p in sorted(kaggle.rglob("train.csv")):
            with open(p, encoding="utf-8") as f:
                head = f.readline().strip()
            if head == "date,target" and (p.parent / "test.csv").exists():
                return p.parent
        raise FileNotFoundError("no train.csv with header 'date,target' (and test.csv beside it) under /kaggle/input")
    try:
        return Path(__file__).resolve().parent.parent / "data"
    except NameError:
        return Path.cwd() / "data"


def load_data(data_dir: Path):
    train = pd.read_csv(data_dir / "train.csv", parse_dates=["date"])
    test = pd.read_csv(data_dir / "test.csv", parse_dates=["date"])
    sample = pd.read_csv(data_dir / "sample_submission.csv")
    assert list(train.columns) == ["date", "target"], list(train.columns)
    assert list(test.columns) == ["date"], list(test.columns)
    assert list(sample.columns) == ["date", "target"], list(sample.columns)
    assert (pd.to_datetime(sample["date"]).values == test["date"].values).all(), "sample dates differ from test dates"
    for name, d in (("train", train), ("test", test)):
        assert (d["date"].diff().dropna() == pd.Timedelta(days=1)).all(), f"{name} dates are not consecutive days"
    assert test["date"].min() == train["date"].max() + pd.Timedelta(days=1) and len(test) == H
    y = train.set_index("date")["target"].astype(float)
    assert y.notna().all() and (y >= 0).all()
    print(f"data: train {len(y)} days {y.index.min().date()} -> {y.index.max().date()}, "
          f"test {len(test)} days {test['date'].min().date()} -> {test['date'].max().date()}")
    return y, test, sample


def anomaly_mask(index: pd.DatetimeIndex) -> pd.Series:
    m = pd.Series(False, index=index)
    for a, b in ANOMALIES:
        m[a:b] = True
    return m


def calendar(dates: pd.DatetimeIndex) -> pd.DataFrame:
    full = pd.date_range(dates.min() - pd.Timedelta(days=15), dates.max() + pd.Timedelta(days=15))
    off = pd.Series((full.dayofweek >= 5) | full.isin(HOLIDAYS), index=full)
    run_id = (off != off.shift()).cumsum()
    run_len = off.groupby(run_id).transform("size")
    long_off = off & (run_len >= 3)
    cal = pd.DataFrame({
        "off": off,
        "long_off": long_off,
        "last_off": off & ~off.shift(-1, fill_value=False),
        "pre_long": ~off & long_off.shift(-1, fill_value=False),
    }).astype(float)
    return cal.loc[dates]


def clean_context(y: pd.Series) -> pd.Series:
    y = y[START:].copy()
    for a, b in ANOMALIES:
        a, b = pd.Timestamp(a), pd.Timestamp(b)
        if a > y.index.max():
            continue
        if b > y.index.max():
            raise ValueError(f"context ends inside anomaly window {a.date()}..{b.date()}")
        before = y[a - pd.Timedelta(weeks=IMPUTE_WEEKS):a - pd.Timedelta(days=1)]
        dow_mean = before.groupby(before.index.dayofweek).mean()
        window = y[a:b].index
        y[window] = dow_mean.reindex(window.dayofweek).values
    return y


def backtest_origins(y: pd.Series) -> list[pd.Timestamp]:
    ends = pd.date_range(START + pd.Timedelta(weeks=8), y.index.max() - pd.Timedelta(days=1), freq="W-WED")
    bad = anomaly_mask(ends)
    return list(ends[~bad.values])


def smape(actual, forecast) -> np.ndarray:
    a, f = np.asarray(actual, float), np.asarray(forecast, float)
    den = np.abs(a) + np.abs(f)
    return np.where(den == 0, 0.0, 200 * np.abs(f - a) / np.where(den == 0, 1, den))
