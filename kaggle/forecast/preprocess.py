from pathlib import Path

import numpy as np
import pandas as pd

VARS = ["tmax", "tmin", "rh", "wind"]
CTX, H = 10, 5
STEP = 45
ORIGIN = pd.Timestamp("1940-01-01")
TEST_START = pd.Timestamp("2015-01-01")
SMOOTH = 15
HALF = int(5.5 * 365.25)
BREAK = pd.Timestamp("2017-01-01")


def get_data_dir():
    root = Path("/kaggle/input")
    if root.exists():
        for p in root.rglob("train.csv"):
            if (p.parent / "test.csv").exists() and (p.parent / "scales.csv").exists():
                return p.parent
    try:
        return Path(__file__).resolve().parent.parent / "data"
    except NameError:
        return Path.cwd() / "data"


def find_input(name):
    root = Path("/kaggle/input")
    if root.exists():
        return next(root.rglob(name))
    return get_data_dir() / "external" / name


def load(d):
    train = pd.read_csv(d / "train.csv")
    test = pd.read_csv(d / "test.csv").sort_values(["window_id", "step"])
    scales = pd.read_csv(d / "scales.csv").set_index("variable")["scale"][VARS].to_numpy()
    return train, test, scales


def season(dates):
    m = dates.month
    return np.select([np.isin(m, [12, 1, 2]), np.isin(m, [3, 4, 5]), np.isin(m, [6, 7, 8, 9])], [0, 1, 2], 3)


def build_panel(train, test):
    stations = sorted(train.station.unique())
    si = {s: i for i, s in enumerate(stations)}
    n_train = train.day_index.max() + 1
    assert (ORIGIN + pd.Timedelta(days=int(n_train))) == TEST_START

    first = test[test.step == -9]
    k = len(first) // len(stations)
    starts = TEST_START + pd.to_timedelta(np.arange(k) * STEP, "D")
    by_doy = dict(zip(starts.dayofyear, starts))
    assert len(by_doy) == k and set(by_doy) == set(first.doy)
    t0 = {w: (by_doy[d] - ORIGIN).days for w, d in zip(first.window_id, first.doy)}

    dates = pd.date_range(ORIGIN, starts[-1] + pd.Timedelta(days=CTX + H - 1))
    X = np.full((len(stations), len(dates), len(VARS)), np.nan)
    X[train.station.map(si).to_numpy(), train.day_index.to_numpy()] = train[VARS].to_numpy()

    day = test.window_id.map(t0).to_numpy() + test.step.to_numpy() + CTX - 1
    assert (dates.dayofyear[day] == test.doy.to_numpy()).all()
    ctx = test.step.to_numpy() <= 0
    X[test.station.map(si).to_numpy()[ctx], day[ctx]] = test.loc[ctx, VARS].to_numpy()

    w = first.window_id.to_numpy()
    windows = pd.DataFrame({"window_id": w, "station": first.station.map(si).to_numpy(),
                            "end": np.array([t0[x] for x in w]) + CTX - 1})
    return X, dates, n_train, windows


def climatology(X, doy, fit):
    C = np.stack([X[:, fit & (doy == k)].mean(1) for k in range(1, 367)], 1)
    p = SMOOTH // 2
    Cp = np.concatenate([C[:, -p:], C, C[:, :p]], 1)
    cs = np.pad(np.cumsum(Cp, 1), ((0, 0), (1, 0), (0, 0)))
    return (cs[:, SMOOTH:] - cs[:, :-SMOOTH]) / SMOOTH


def offsets(A, group):
    T = A.shape[1]
    i = np.arange(T)
    lo, hi = np.clip(i - HALF, 0, T), np.clip(i + HALF + 1, 0, T)
    L = np.zeros_like(A)
    for g in np.unique(group):
        m = (group == g)[None, :, None] & ~np.isnan(A)
        cv = np.pad(np.cumsum(np.where(m, A, 0), 1), ((0, 0), (1, 0), (0, 0)))
        cn = np.pad(np.cumsum(m, 1), ((0, 0), (1, 0), (0, 0)))
        mean = (cv[:, hi] - cv[:, lo]) / np.maximum(cn[:, hi] - cn[:, lo], 1)
        L[:, group == g] = mean[:, group == g]
    return L


class Frame:
    def __init__(self, X, dates, fit):
        self.doy = dates.dayofyear.to_numpy()
        self.C = climatology(X, self.doy, fit)
        A = X - self.C[:, self.doy - 1]
        self.L = offsets(A, season(dates) + 4 * (dates >= BREAK))
        self.base = self.C[:, self.doy - 1] + self.L
        self.Z = X - self.base
        self.X = X

    def features(self, st, ends):
        own = self.Z[st[:, None], ends[:, None] + np.arange(1 - CTX, 1)].reshape(len(st), -1)
        recent = self.Z[:, ends[:, None] + np.arange(-2, 1)]
        others = np.concatenate([recent[:, :, -1], recent.mean(2)], 2).transpose(1, 0, 2).reshape(len(st), -1)
        ahead = self.base[st[:, None], ends[:, None] + np.arange(1, H + 1)].reshape(len(st), -1)
        ang = 2 * np.pi * self.doy[ends] / 365.25
        return np.column_stack([st, np.sin(ang), np.cos(ang), own, others, ahead, self.L[st, ends]]).astype(np.float32)

    def targets(self, st, ends):
        return self.Z[st[:, None], ends[:, None] + np.arange(1, H + 1)]

    def to_values(self, st, ends, anomaly):
        return self.base[st[:, None], ends[:, None] + np.arange(1, H + 1)] + anomaly
