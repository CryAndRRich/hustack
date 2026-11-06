import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from nwp import Stack, load_nwp
from preprocess import CTX, H, STEP, VARS, Frame, build_panel, climatology, find_input, get_data_dir, load

VERSION = "v4.3"
NWP_FILES = ["nwp_daily.csv", "hres_daily.csv", "eod_daily.csv", "gefs_daily.csv"]
HO_START = pd.Timestamp("2005-01-01")
STRIDE = 2
ROUNDS = 400
CHECK = [100, 200, 300, 400]
PARAMS = {"objective": "l1", "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 100,
          "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1, "seed": 0}


def out_dir():
    return Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")


def score(pred, truth, scales):
    per = (np.abs(pred - truth) / scales).mean((0, 1))
    return per.mean(), dict(zip(VARS, per.round(4)))


def train_rows(n_stations, first_end, last_end):
    ends = np.arange(first_end, last_end + 1, STRIDE)
    return np.repeat(np.arange(n_stations), len(ends)), np.tile(ends, n_stations)


def fit_predict(fr, st, ends, st_p, ends_p):
    F, Y, Fp = fr.features(st, ends), fr.targets(st, ends), fr.features(st_p, ends_p)
    out = np.zeros((len(st_p), H, len(VARS), len(CHECK)))
    for h in range(H):
        for v in range(len(VARS)):
            m = lgb.train(PARAMS, lgb.Dataset(F, Y[:, h, v], categorical_feature=[0]), ROUNDS)
            for j, k in enumerate(CHECK):
                out[:, h, v, j] = m.predict(Fp, num_iteration=k)
    return out


def validate(X, dates, n_train, scales):
    ho = (HO_START - dates[0]).days
    starts = np.arange(ho, n_train - CTX - H + 1, STEP)
    ends = starts + CTX - 1
    keep = np.zeros(n_train, bool)
    for s in starts:
        keep[s:s + CTX] = True
    Xv = X[:, :n_train].copy()
    Xv[:, ho:][:, ~keep[ho:]] = np.nan
    fit = np.arange(n_train) < ho
    fr = Frame(Xv, dates[:n_train], fit)

    S = X.shape[0]
    st_p, ends_p = np.repeat(np.arange(S), len(ends)), np.tile(ends, S)
    truth = X[st_p[:, None], ends_p[:, None] + np.arange(1, H + 1)]
    doy = fr.doy[ends_p[:, None] + np.arange(1, H + 1)]
    C20 = climatology(Xv, fr.doy, fit & (np.arange(n_train) >= ho - 20 * 365))
    ref = C20[st_p[:, None], doy - 1]
    for name, p in [("starter climatology", ref), ("climatology + period offset", fr.to_values(st_p, ends_p, 0)),
                    ("persistence", np.repeat(X[st_p, ends_p][:, None], H, 1))]:
        print(f"  {name:28s} {score(p, truth, scales)}")

    st, e = train_rows(S, CTX - 1, ho - H - 1)
    pred = fit_predict(fr, st, e, st_p, ends_p)
    for j, k in enumerate(CHECK):
        p = fr.to_values(st_p, ends_p, pred[..., j])
        total, per = score(p, truth, scales)
        ratio = (np.abs(p - truth).mean((0, 1)) / np.abs(ref - truth).mean((0, 1))).mean()
        print(f"  GBM {k:4d} rounds   scaled {total:.4f}  vs starter clim {ratio:.4f}  {per}")
    return pred[..., -1], truth


def run(out):
    t0 = time.time()
    train, test, scales = load(get_data_dir())
    X, dates, n_train, windows = build_panel(train, test)
    print(f"panel {X.shape}, {len(windows)} test windows, train days {n_train}")

    print("validation: windows of 2005-2014 on a 45-day chain, fit before 2005")
    val_pred, val_truth = validate(X, dates, n_train, scales)
    print(f"  ({time.time() - t0:.0f}s)")

    fit = np.arange(len(dates)) < n_train
    fr = Frame(X, dates, fit)
    st, e = train_rows(X.shape[0], CTX - 1, n_train - H - 1)
    st_p, ends_p = windows.station.to_numpy(), windows.end.to_numpy()
    days = ends_p[:, None] + np.arange(1 - CTX, 1)
    C20 = climatology(X, fr.doy, fit & (np.arange(len(dates)) >= n_train - 20 * 365))
    print("test context days (known; offset in-sample):")
    for name, p in [("starter climatology", C20[st_p[:, None], fr.doy[days] - 1]),
                    ("climatology + period offset", fr.base[st_p[:, None], days])]:
        print(f"  {name:28s} {score(p, X[st_p[:, None], days], scales)}")
    G = fit_predict(fr, st, e, st_p, ends_p)[..., -1]
    N, t0 = load_nwp([find_input(f) for f in NWP_FILES], dates, X.shape[0])
    stack = Stack(fr, N, t0, windows, n_train, dates)
    print(f"NWP stack: {len(stack.coef)} (variable, lead, model set) fits on {len(stack.pairs[0])} context pairs; CV:")
    stack.cv_report(scales)
    pred = fr.to_values(st_p, ends_p, stack.combine(st_p, ends_p, G))

    ids = (np.repeat(windows.window_id.to_numpy(), H * len(VARS)) + "_"
           + np.tile(np.repeat(np.arange(1, H + 1), len(VARS)), len(windows)).astype(str) + "_"
           + np.tile(VARS, len(windows) * H))
    sub = pd.DataFrame({"id": ids, "value": (pred / scales).ravel()}).sort_values("id")
    sample = pd.read_csv(get_data_dir() / "sample_submission.csv")
    assert len(sub) == len(sample) and set(sub.id) == set(sample.id) and np.isfinite(sub.value).all()
    sub.to_csv(out / "submission.csv", index=False)
    (out / "extra").mkdir(exist_ok=True)
    np.savez(out / "extra" / f"train_{VERSION}.npz", val_pred=val_pred, val_truth=val_truth, test_pred=pred, test_gbm=G,
             window_id=windows.window_id.to_numpy())
    print(f"done in {time.time() - t0:.0f}s")
    print(f"SUBMIT: {out / 'submission.csv'}")


if __name__ == "__main__":
    run(out_dir())
