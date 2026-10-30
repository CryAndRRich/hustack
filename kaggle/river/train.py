import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from explore import report
from preprocess import NWM, TARGET, features, load

VAL_YEARS = [2007, 2010, 2015, 2016]
FIRST = pd.Timestamp("1981-01-01")
LEADS = ["Q_pred_t", "Q_pred_t1", "Q_pred_t2", "Q_pred_t3"]
PARAMS = dict(objective="regression", learning_rate=0.03, num_leaves=63, min_child_samples=100,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=42)
ROUNDS, STOP = 5000, 200
OUT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - b) ** 2)))


def shifted(v, basin):
    s = pd.Series(v, index=basin.index)
    cols = [s.groupby(basin).shift(-k).groupby(basin).ffill() for k in range(len(LEADS))]
    return np.column_stack(cols)


def run():
    t0 = time.time()
    df, sub = load()
    report(df)
    X = features(df)
    print(f"features {X.shape[1]} ({time.time() - t0:.0f}s)")

    y = df[TARGET].values
    year = df.date.dt.year.values
    use = ((df.part == "train") & df[TARGET].notna() & (df.date >= FIRST)).values
    print("training hours per basin: " + ", ".join(f"{b} {n}" for b, n in df[use].basin.value_counts().sort_index().items()))
    rows, iters, imp = [], [], pd.Series(0.0, index=X.columns)
    for vy in VAL_YEARS:
        a, b = np.flatnonzero(use & (year != vy)), np.flatnonzero(use & (year == vy))
        m = lgb.train(PARAMS, lgb.Dataset(X.iloc[a], y[a]), ROUNDS, valid_sets=[lgb.Dataset(X.iloc[b], y[b])],
                      callbacks=[lgb.early_stopping(STOP, verbose=False)])
        p = np.clip(m.predict(X.iloc[b], num_iteration=m.best_iteration), 0, None)
        iters.append(m.best_iteration)
        imp += pd.Series(m.feature_importance("gain"), index=X.columns)
        yb, basin = y[b], df.basin.iloc[b]
        top = yb >= np.quantile(yb, 0.99)
        ty, tp = shifted(yb, basin), shifted(p, basin)
        rows.append(dict(year=vy, iter=m.best_iteration, rmse_zero=rmse(yb, 0), rmse_nwm=rmse(yb, X[NWM].values[b]),
                         rmse=rmse(yb, p),
                         rmse_leads=rmse(ty, tp), nse=1 - np.sum((yb - p) ** 2) / np.sum((yb - yb.mean()) ** 2),
                         rmse_top1=rmse(yb[top], p[top]), peak_ratio=p[top].mean() / yb[top].mean(),
                         sum_ratio=p.sum() / yb.sum()))
        print(f"year {vy}: " + ", ".join(f"{k} {v:.5f}" for k, v in rows[-1].items() if k != "year")
              + f" ({time.time() - t0:.0f}s)")
        print("  per basin rmse model/zero: " + ", ".join(
            f"{k} {rmse(yb[g], p[g]):.4f}/{rmse(yb[g], 0):.4f}" for k, g in basin.groupby(basin).indices.items()))
    R = pd.DataFrame(rows)
    print(R.round(5).to_string(index=False))
    print(f"CV mean: rmse {R.rmse.mean():.5f} vs zero {R.rmse_zero.mean():.5f}, NSE {R.nse.mean():.3f}; "
          f"V1/V2 folds (2007/2010/2015) rmse {R.rmse[R.year != 2016].mean():.5f}")
    print("top features (gain): " + ", ".join(f"{c} {v / imp.sum():.3f}" for c, v in imp.nlargest(20).items()))

    rounds = int(np.median(iters))
    a = np.flatnonzero(use)
    m = lgb.train(PARAMS, lgb.Dataset(X.iloc[a], y[a]), rounds)
    te = np.flatnonzero((df.part == "test").values)
    pred = np.clip(m.predict(X.iloc[te]), 0, None)
    T = df.iloc[te][["basin", "date"]].copy()
    T[LEADS] = shifted(pred, T.basin)
    T = T[T.date.dt.year == 2018]
    T["Unique_ID"] = T.date.dt.strftime("%Y-%m-%d %H:%M:%S") + "_" + T.basin
    out = sub[["Unique_ID"]].merge(T[["Unique_ID"] + LEADS], on="Unique_ID", how="left")
    assert len(out) == len(sub) and out[LEADS].notna().all().all(), "submission ids did not all match"
    print("2018 prediction: mean by month " + ", ".join(
        f"{k} {v:.4f}" for k, v in T.groupby(T.date.dt.month).Q_pred_t.mean().items()))
    print("2018 prediction max per basin: " + ", ".join(f"{k} {v:.3f}" for k, v in T.groupby("basin").Q_pred_t.max().items()))
    out.to_csv(OUT / "submission.csv", index=False)
    print(f"SUBMIT: submission.csv  (V2.1: V2 + NWM v2.1 features, {rounds} rounds, CV rmse {R.rmse.mean():.5f}) "
          f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    run()
