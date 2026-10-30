import functools
import importlib
import os
import shutil
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from preprocess import (COVARIATES, H, anomaly_mask, backtest_origins, calendar, clean_context, get_data_dir,
                        load_data, smape)

VERSION = "v1.2"
AUTO_SMOKE = True
SMOKE_ORIGINS = 2
SUBMIT_ARM = "trend_lb"
TREND_WEEKS = 8
LB_FIT = {"c": -0.034, "b": -0.451, "gamma": 0.376}
REF_ARM = "wk4"
MARGIN = 0.5
MIN_WIN_SHARE = 0.6
RECENT_FROM = pd.Timestamp("2021-10-06")
WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
PINS = {"chronos": "chronos-forecasting==2.3.2", "timesfm3": "timesfm==3.0.2"}


def pip_import(module: str, attr: str):
    try:
        return getattr(importlib.import_module(module), attr)
    except (ImportError, AttributeError):
        print(f"{module}.{attr} not available — pip install {PINS[module]}")
        r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", PINS[module]], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"pip install {PINS[module]} failed — turn the notebook's Internet ON\n" + r.stderr[-2000:])
        for name in [m for m in sys.modules if m == module or m.startswith(module + ".")]:
            del sys.modules[name]
        importlib.invalidate_caches()
        return getattr(importlib.import_module(module), attr)


@functools.lru_cache(maxsize=None)
def chronos2():
    import torch
    torch.set_num_threads(os.cpu_count())
    pipe = pip_import("chronos", "Chronos2Pipeline").from_pretrained("amazon/chronos-2", device_map="cpu")
    print(f"chronos-2 loaded, torch threads {torch.get_num_threads()}")
    return pipe


@functools.lru_cache(maxsize=None)
def timesfm3():
    model = pip_import("timesfm3", "TimesFM3Forecaster").from_pretrained(
        "google/timesfm-3.0-pytorch", device="cpu", per_core_batch_size=32)
    print("timesfm-3.0 loaded")
    return model


def future_dates(ctx: pd.Series) -> pd.DatetimeIndex:
    return pd.date_range(ctx.index.max() + pd.Timedelta(days=1), periods=H)


def weekday_mean(contexts, weeks):
    out = []
    for ctx in contexts:
        last = ctx.iloc[-7 * weeks:]
        means = last.groupby(last.index.dayofweek).mean()
        out.append(means.reindex(future_dates(ctx).dayofweek).values)
    return np.array(out)


def trend_parts(ctx):
    n = min(len(ctx) // 7, TREND_WEEKS) * 7
    last = ctx.iloc[-n:]
    df = pd.DataFrame({"y": last.values, "block": np.arange(n) // 7, "dow": last.index.dayofweek,
                       "t": np.arange(n) - (n - 1)})[~anomaly_mask(last.index).values]
    days = df.groupby("block")["y"].transform("size")
    full = df[days == 7]
    prof = (full["y"] / full.groupby("block")["y"].transform("mean")).groupby(full["dow"]).mean()
    prof /= prof.mean()
    df["p"] = df["dow"].map(prof)
    wk = df[days >= 4].groupby("block").agg(y=("y", "mean"), p=("p", "mean"), t=("t", "mean"))
    level = wk["y"] / wk["p"]
    slope, icpt = np.polyfit(wk["t"], level, 1) if len(wk) >= 3 else (0.0, level.mean())
    return icpt + slope * np.arange(1, H + 1), prof.reindex(future_dates(ctx).dayofweek).values


def trend_profile(contexts):
    return np.array([line * prof for line, prof in map(trend_parts, contexts)])


def trend_lb(contexts):
    tilt = (np.arange(H) - (H - 1) / 2) / (H - 1)
    out = []
    for ctx in contexts:
        line, prof = trend_parts(ctx)
        logp = np.log(prof) - np.log(prof[:7]).mean()
        out.append(line * prof * np.exp(LB_FIT["c"] + LB_FIT["b"] * tilt + (LB_FIT["gamma"] - 1) * logp))
    return np.array(out)


def holt_winters(contexts):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    out = []
    for ctx in contexts:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = ExponentialSmoothing(ctx.values, trend="add", damped_trend=True, seasonal="mul",
                                       seasonal_periods=7, initialization_method="estimated").fit()
        out.append(fit.forecast(H))
    return np.array(out)


def chronos2_arm(contexts, cov):
    if cov:
        inputs = []
        for ctx in contexts:
            cal_p, cal_f = calendar(ctx.index), calendar(future_dates(ctx))
            inputs.append({"target": ctx.values.astype(np.float32),
                           "past_covariates": {c: cal_p[c].values for c in COVARIATES},
                           "future_covariates": {c: cal_f[c].values for c in COVARIATES}})
    else:
        inputs = [ctx.values.astype(np.float32) for ctx in contexts]
    _, median = chronos2().predict_quantiles(inputs, prediction_length=H, quantile_levels=[0.5])
    return np.array([m[0].numpy() for m in median])


def timesfm3_arm(contexts, cov):
    pf = None
    if cov:
        pf = [calendar(ctx.index.append(future_dates(ctx)))[COVARIATES].values.T.astype(np.float32) for ctx in contexts]
    outs = timesfm3().predict_batch(contexts=[ctx.values.astype(np.float32) for ctx in contexts], horizon=H,
                                    past_future_covariates=pf, use_symmetric_averaging=True, make_positive=True)
    return np.array([o.forecast for o in outs])


ARMS = {
    "wk1": lambda c: weekday_mean(c, 1),
    "wk2": lambda c: weekday_mean(c, 2),
    "wk4": lambda c: weekday_mean(c, 4),
    "wk8": lambda c: weekday_mean(c, 8),
    "trend": trend_profile,
    "trend_lb": trend_lb,
    "holt_winters": holt_winters,
    "chronos2": lambda c: chronos2_arm(c, cov=False),
    "chronos2_cov": lambda c: chronos2_arm(c, cov=True),
    "timesfm3": lambda c: timesfm3_arm(c, cov=False),
    "timesfm3_cov": lambda c: timesfm3_arm(c, cov=True),
}


def run_arms(contexts):
    preds = {}
    for name, fn in ARMS.items():
        t0 = time.time()
        try:
            p = np.asarray(fn(contexts), float)
            assert p.shape == (len(contexts), H), p.shape
            assert np.isfinite(p).all(), "non-finite forecast"
            preds[name] = np.maximum(p, 1.0)
            print(f"  {name:<13} {time.time() - t0:6.1f}s")
        except Exception as e:
            print(f"  {name:<13} FAILED after {time.time() - t0:.1f}s: {type(e).__name__}: {str(e)[:300]}")
    assert REF_ARM in preds, f"reference arm {REF_ARM} failed"
    return preds


def score(y, origins, preds):
    bad = anomaly_mask(y.index)
    rows = []
    for i, end in enumerate(origins):
        dates = pd.date_range(end + pd.Timedelta(days=1), periods=H)
        keep = dates.isin(y.index)
        keep[keep] = ~bad.reindex(dates[keep]).values
        actual = y.reindex(dates[keep]).values
        for arm, p in preds.items():
            f = p[i][keep]
            rows.append(pd.DataFrame({"origin": end, "date": dates[keep], "arm": arm, "actual": actual,
                                      "forecast": f, "smape": smape(actual, f)}))
    return pd.concat(rows, ignore_index=True)


def summarise(bt):
    per_origin = bt.pivot_table(index="origin", columns="arm", values="smape", aggfunc="mean")
    recent = bt[bt["origin"] >= RECENT_FROM]
    summary = pd.DataFrame({
        "pooled": bt.groupby("arm")["smape"].mean(),
        "recent": recent.groupby("arm")["smape"].mean(),
        "origin_mean": per_origin.mean(),
        "win_vs_ref": (per_origin.lt(per_origin[REF_ARM], axis=0)).mean(),
    }).sort_values("pooled")
    return summary, per_origin


def choose(summary):
    best = summary.index[0]
    gain = summary.loc[REF_ARM, "pooled"] - summary.loc[best, "pooled"]
    win = summary.loc[best, "win_vs_ref"]
    passed = best != REF_ARM and gain >= MARGIN and win >= MIN_WIN_SHARE
    print(f"gate: best {best} gains {gain:+.3f} SMAPE pooled over {REF_ARM}, beats it on {win:.0%} of origins "
          f"(need >= {MARGIN} and >= {MIN_WIN_SHARE:.0%}) -> {'PASS' if passed else 'keep ' + REF_ARM}")
    return best if passed else REF_ARM


def write_submission(path, test, sample, forecast):
    sub = pd.DataFrame({"date": test["date"].dt.strftime("%Y-%m-%d"), "target": forecast})
    assert len(sub) == len(sample) and list(sub.columns) == list(sample.columns)
    assert (sub["date"].values == sample["date"].values).all(), "submission dates differ from sample"
    assert np.isfinite(sub["target"]).all() and (sub["target"] > 0).all()
    sub.to_csv(path, index=False)


def run(n_origins, out_dir, tag):
    t0 = time.time()
    print(f"\n===== {tag} =====")
    y, test, sample = load_data(get_data_dir())
    origins = backtest_origins(y)
    if n_origins is not None:
        origins = origins[-n_origins:]
    contexts = [clean_context(y[:end]) for end in origins] + [clean_context(y)]
    print(f"backtest: {len(origins)} origins {origins[0].date()} -> {origins[-1].date()}, "
          f"context from {contexts[-1].index.min().date()}, final context {len(contexts[-1])} days")
    preds = run_arms(contexts)
    bt = score(y, origins, {k: v[:-1] for k, v in preds.items()})
    summary, per_origin = summarise(bt)
    pd.set_option("display.width", 250)
    print("\nbacktest SMAPE (pooled = all scored origin-days; recent = origins from "
          f"{RECENT_FROM.date()}; win_vs_ref = share of origins beating {REF_ARM}):")
    print(summary.round(3).to_string())
    print("\nper-origin mean SMAPE:")
    print(per_origin[summary.index].round(2).to_string())
    gated = choose(summary)
    assert SUBMIT_ARM in preds, f"submission arm {SUBMIT_ARM} failed"
    chosen = SUBMIT_ARM
    print(f"submitting {chosen} (pre-registered; the gate's pick was {gated})")

    test_arms = pd.DataFrame({k: v[-1] for k, v in preds.items()}, index=test["date"].dt.date)
    print("\ntest forecasts per arm:")
    print(test_arms[summary.index].round(0).to_string())
    print("December mean per arm:", test_arms[summary.index].mean().round(0).to_dict())
    out_dir.mkdir(parents=True, exist_ok=True)
    extra = out_dir / "extra"
    extra.mkdir(exist_ok=True)
    path = out_dir / "submission.csv"
    write_submission(path, test, sample, preds[chosen][-1])
    bt.to_csv(extra / f"backtest_{VERSION}.csv", index=False)
    summary.to_csv(extra / f"backtest_summary_{VERSION}.csv")
    test_arms.to_csv(extra / f"test_arms_{VERSION}.csv")
    print(f"{tag} done in {time.time() - t0:.0f}s")
    return path, chosen


def main():
    if AUTO_SMOKE:
        smoke_dir = WORK / "smoke_tmp"
        run(SMOKE_ORIGINS, smoke_dir, "SMOKE")
        shutil.rmtree(smoke_dir, ignore_errors=True)
        print("smoke pass OK — outputs discarded")
    path, chosen = run(None, WORK, "FULL")
    print(f"\n{VERSION} chosen arm: {chosen}")
    print(f"SUBMIT: {path}")


if __name__ == "__main__":
    main()
