import json
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

from preprocess import CLIP, SAW, design, features, load, load_correction

CFG = {"block": 10000, "trees": 2000, "cat_iters": 3000, "mlp_iter": 300}
SMOKE = {"block": 25000, "trees": 20, "cat_iters": 20, "mlp_iter": 5}
OUT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")


def ols(X, y):
    m = y < CLIP
    return np.linalg.lstsq(X[m], y[m], rcond=None)[0]


class SplineHour:
    def __init__(self):
        self.spline = SplineTransformer(n_knots=12, degree=3)
        self.ridge = Ridge(alpha=1e-3)

    def _x(self, f, fit=False):
        s = self.spline.fit_transform(f[["temperature_c"]]) if fit else self.spline.transform(f[["temperature_c"]])
        h = np.eye(24)[f.hour.values]
        rest = f[["is_holiday", "weekend", "humidity_pct"]].values
        day = f.hour.between(9, 18).values[:, None]
        return np.hstack([(h[:, :, None] * s[:, None, :]).reshape(len(f), -1), rest, rest[:, 1:2] * day])

    def fit(self, f, y):
        self.ridge.fit(self._x(f, fit=True), y)
        return self

    def predict(self, f):
        return self.ridge.predict(self._x(f))


def families(cfg):
    import lightgbm as lgb
    import xgboost as xgb
    from catboost import CatBoostRegressor

    n = cfg["trees"]
    return {
        "lgbm": lambda: lgb.LGBMRegressor(n_estimators=n, learning_rate=0.03, num_leaves=63, subsample=0.8,
                                          subsample_freq=1, colsample_bytree=0.9, verbose=-1),
        "cat": lambda: CatBoostRegressor(iterations=cfg["cat_iters"], learning_rate=0.05, depth=8, verbose=0),
        "xgb": lambda: xgb.XGBRegressor(n_estimators=n, learning_rate=0.03, max_depth=8, subsample=0.8,
                                        colsample_bytree=0.9, tree_method="hist"),
        "knn": lambda: make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=10, weights="distance")),
        "mlp": lambda: make_pipeline(StandardScaler(), MLPRegressor((256, 256), early_stopping=True,
                                                                    max_iter=cfg["mlp_iter"], random_state=0)),
        "spline": SplineHour,
    }


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def run(cfg, smoke):
    t0 = time.time()
    tr, te = load()
    pc = load_correction(te)
    X, names = design(tr)
    Xt, _ = design(te)
    saw = names.index("saw")
    y = tr.load_mw.values.astype(float)

    beta = ols(X, y)
    Xt[:, saw] = (SAW - 1) / 2
    p = np.minimum(Xt @ beta, CLIP)
    gap = np.abs(p - pc.v11.values).max()
    print(f"ols test prediction vs scored V1.1: max |diff| {gap:.4f}")
    assert gap < 0.05, "the OLS fit does not reproduce V1.1"
    best = np.minimum(p + pc["corr"].values, CLIP)

    m = y < CLIP
    yc = (y - beta[saw] * (tr.id.values % SAW - (SAW - 1) / 2))[m]
    F, Ft = features(tr)[m].reset_index(drop=True), features(te)
    fold = tr.id.values[m] // cfg["block"]
    ref = np.zeros(m.sum())
    Xc = X[m].copy()
    Xc[:, saw] = (SAW - 1) / 2
    for f in np.unique(fold):
        va = fold == f
        ref[va] = Xc[va] @ ols(X[m][~va], y[m][~va])
    cv = {"ols": rmse(ref, yc)}
    print(f"ols            block cv {cv['ols']:.4f} (rounding floor 0.289)")

    probes, oof = {}, {"id": tr.id.values[m], "y_clean": yc, "ols": ref}
    for k, make in families(cfg).items():
        t1 = time.time()
        try:
            o = np.zeros(len(yc))
            for f in np.unique(fold):
                va = fold == f
                o[va] = make().fit(F[~va], yc[~va]).predict(F[va])
            v = make().fit(F, yc).predict(Ft)
        except Exception as e:
            print(f"{k:14s} FAILED: {e!r}")
            continue
        cv[k], oof[k] = rmse(o, yc), o
        d = np.minimum(v, CLIP) - p
        probes[k] = np.minimum(best + d, CLIP)
        print(f"{k:14s} block cv {cv[k]:.4f} | test: rms(family - ols) {rmse(d, 0):.3f} | {time.time() - t1:.0f}s")

    if smoke:
        print(f"smoke ok ({time.time() - t0:.0f}s)")
        return
    sub = pd.DataFrame({"id": te.id, "load_mw": best})
    assert len(sub) == len(te) and np.isfinite(sub.load_mw).all() and sub.load_mw.max() <= CLIP
    sub.to_csv(OUT / "submission.csv", index=False)
    with zipfile.ZipFile(OUT / "extra_v2.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for k, v in probes.items():
            z.writestr(f"probe_{k}.csv", pd.DataFrame({"id": te.id, "load_mw": v}).to_csv(index=False))
        z.writestr("oof.csv", pd.DataFrame(oof).to_csv(index=False))
        z.writestr("cv.json", json.dumps(cv, indent=1))
    print(f"families written: {list(probes)}; {time.time() - t0:.0f}s")
    print(f"SUBMIT: {OUT / 'submission.csv'}")


def main():
    run({**CFG, **SMOKE}, smoke=True)
    run(CFG, smoke=False)


if __name__ == "__main__":
    main()
