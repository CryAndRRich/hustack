import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from gp import block_loo, fit_gp, predict_gp
from preprocess import OUT, describe, load, load_sites, log, nearest, shift_report

N_REPEATS = 60
N_HOLD = 40
SEED = 0
DIST_BINS = np.array([0.03, 0.12, 0.2])
V1_FACTORS = ("Rock", "Landuse")
LAG_METALS = ["Ni", "Cr", "Zn"]
LAG = ["co_lag", "co_lag_sd", "lag_dist"] + [f"{m.lower()}_lag" for m in LAG_METALS]
TABPFN_EST = 8
V1 = "gp V1"


def setup():
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tabpfn"], check=True)
    if Path("/kaggle/working").exists():
        from kaggle_secrets import UserSecretsClient
        secrets = UserSecretsClient()
        for key in ("HF_TOKEN", "TABPFN_TOKEN"):
            try:
                os.environ[key] = secrets.get_secret(key)
            except Exception:
                log(f"secret {key} not available")
    os.environ["TABPFN_NO_BROWSER"] = "1"
    if not os.environ.get("TABPFN_TOKEN"):
        os.environ["TABPFN_MODEL_VERSION"] = "v2"
        log("WARNING: no TABPFN_TOKEN secret -> TabPFN v2 (ungated) instead of the default")
    import torch
    from importlib.metadata import version
    assert torch.cuda.is_available(), "GPU accelerator required"
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 3))
    p = tabpfn_predict(pd.DataFrame(x[:30]), x[:30, 0] * 2, pd.DataFrame(x[30:]))
    assert np.isfinite(p).all() and np.corrcoef(p, x[30:, 0])[0, 1] > 0.8, "TabPFN smoke fit failed"
    log(f"tabpfn {version('tabpfn')} torch {torch.__version__} gpu {torch.cuda.get_device_name(0)} model version "
        f"{os.environ.get('TABPFN_MODEL_VERSION', 'default')}; smoke fit ok")


def tabpfn_predict(xtr, y, xte):
    from tabpfn import TabPFNRegressor
    m = TabPFNRegressor(n_estimators=TABPFN_EST, device="cuda", ignore_pretraining_limits=True, random_state=SEED)
    return m.fit(xtr.to_numpy(float), y).predict(xte.to_numpy(float))


def gbdt_predict(xtr, y, xte):
    m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=8, min_samples_leaf=10,
                                      random_state=SEED)
    return m.fit(xtr.to_numpy(float), y).predict(xte.to_numpy(float))


def lag_features(tr, te):
    xy_tr, xy_te = tr[["Xloc", "Yloc"]].to_numpy(float), te[["Xloc", "Yloc"]].to_numpy(float)
    single = np.arange(len(tr))
    m = fit_gp(tr, tr["Co"].to_numpy(float), V1_FACTORS)
    ltr, lte = {}, {}
    ltr["co_lag"], ltr["co_lag_sd"] = block_loo(m, single)
    lte["co_lag"], lte["co_lag_sd"] = predict_gp(m, te)
    ltr["lag_dist"], lte["lag_dist"] = nearest(xy_tr, xy_tr, same=True), nearest(xy_te, xy_tr)
    for metal in LAG_METALS:
        mm = fit_gp(tr, np.log(tr[metal].to_numpy(float)), ())
        ltr[f"{metal.lower()}_lag"], lte[f"{metal.lower()}_lag"] = block_loo(mm, single)[0], predict_gp(mm, te)[0]
    return pd.DataFrame(ltr, index=tr.index), pd.DataFrame(lte, index=te.index), m


def candidate_specs(blocks):
    base = blocks["base"]
    specs, seen = {}, set()
    for name, cols in (("tabpfn base", base), ("tabpfn kpr", base + LAG),
                       ("tabpfn kpr+terrain", base + LAG + blocks["terrain"]),
                       ("tabpfn kpr+all", base + LAG + blocks["terrain"] + blocks["ext"] + blocks["aef"])):
        if tuple(cols) in seen:
            log(f"{name}: same features as an earlier candidate (missing block) -> dropped")
            continue
        seen.add(tuple(cols))
        specs[name] = cols
    return specs


def predict_all(tr, te, specs):
    ltr, lte, m = lag_features(tr, te)
    trf, tef = pd.concat([tr, ltr], axis=1), pd.concat([te, lte], axis=1)
    y = tr["Co"].to_numpy(float)
    preds = {V1: lte["co_lag"].to_numpy()}
    for name, cols in specs.items():
        preds[name] = tabpfn_predict(trf[cols], y, tef[cols])
    all_cols = list(specs.values())[-1]
    preds["gbdt kpr+all"] = gbdt_predict(trf[all_cols], y, tef[all_cols])
    kpr = [n for n in specs if "kpr" in n]
    preds["mean of tabpfn kpr"] = np.mean([preds[n] for n in kpr], 0)
    preds["tabpfn kpr+all & V1"] = (preds[kpr[-1]] + preds[V1]) / 2
    return preds, m, tef


def holdout(d, max_km, rng):
    held = np.zeros(len(d), bool)
    for i in rng.permutation(len(d)):
        if held.sum() >= N_HOLD:
            break
        trial = held.copy()
        trial[i] = True
        if d[np.ix_(trial, ~trial)].min(1).max() <= max_km:
            held = trial
    return held


def quantiles(d):
    return np.round(np.quantile(d, [.1, .25, .5, .75, .9, 1]), 3).tolist()


def distance_weights(d_test, d_cv):
    p_test = np.bincount(np.digitize(d_test, DIST_BINS), minlength=4) / len(d_test)
    p_cv = np.bincount(np.digitize(d_cv, DIST_BINS), minlength=4) / len(d_cv)
    for b in np.flatnonzero((p_test > 0) & (p_cv == 0)):
        to = min(np.flatnonzero(p_cv > 0), key=lambda k: abs(k - b))
        log(f"distance bin {b} never occurs in CV: its test share {p_test[b]:.3f} is merged into bin {to}")
        p_test[to] += p_test[b]
        p_test[b] = 0.0
    w_bin = np.divide(p_test, p_cv, out=np.zeros(4), where=p_cv > 0)
    log("nearest-train distance bins (km) [<0.03, 0.03-0.12, 0.12-0.2, >=0.2 (at most the test max)]: test share "
        f"{np.round(p_test, 3).tolist()}, CV share {np.round(p_cv, 3).tolist()}, weight {np.round(w_bin, 3).tolist()}")
    return w_bin


def summarise(mse):
    best = mse.mean().idxmin()
    diff = mse.sub(mse[best], axis=0)
    return best, pd.DataFrame({"cv_mse": mse.mean(), "se": mse.std() / np.sqrt(len(mse)),
                               "minus_best": diff.mean(), "paired_se": diff.std() / np.sqrt(len(mse))})


if __name__ == "__main__":
    setup()
    train, test = load()
    describe(train, test)
    train_all, test_all, blocks = load_sites(train, test)
    shift_report(train_all, test_all, blocks)
    specs = candidate_specs(blocks)
    assert (train_all["split"] == "train").all(), "V3.1 expects competition training rows only"
    xy = train_all[["Xloc", "Yloc"]].to_numpy(float)
    d_all = np.sqrt(((xy[:, None] - xy[None]) ** 2).sum(-1))
    d_test = nearest(test_all[["Xloc", "Yloc"]].to_numpy(float), xy)
    max_km = d_test.max() + 1e-3
    log(f"nearest-train distance quantiles 10/25/50/75/90/100 % (km): test {quantiles(d_test)}, train leave-one-out "
        f"{quantiles(nearest(xy, xy, same=True))}; each repeat holds out up to {N_HOLD} random sites, each keeping a "
        f"training site within {max_km:.3f} km (the test max)")
    recs = []
    for r in range(N_REPEATS):
        t = time.time()
        held = holdout(d_all, max_km, np.random.default_rng(SEED + r))
        tr, te = train_all[~held], train_all[held]
        preds, _, _ = predict_all(tr, te, specs)
        recs.append(pd.DataFrame({"repeat": r, "site": te.index, "y": te["Co"].to_numpy(),
                                  "d": nearest(te[["Xloc", "Yloc"]].to_numpy(float), tr[["Xloc", "Yloc"]].to_numpy(float)),
                                  **preds}))
        if r == 0:
            mse0 = recs[0][list(preds)].sub(recs[0]["y"], axis=0).pow(2).mean()
            log(f"repeat 0 took {time.time() - t:.1f} s ({len(te)} held-out sites): "
                + ", ".join(f"{n} {v:.3f}" for n, v in mse0.items()))
        elif (r + 1) % 10 == 0:
            run = pd.concat(recs)
            log(f"{r + 1}/{N_REPEATS} repeats; running plain MSE: " + ", ".join(
                f"{n} {v:.3f}" for n, v in run[list(preds)].sub(run["y"], axis=0).pow(2).mean().sort_values().items()))
    rec = pd.concat(recs, ignore_index=True)
    names = [c for c in rec.columns if c not in ("repeat", "site", "y", "d", "bin")]
    assert rec["d"].max() <= max_km, "a CV site sits further from training than any test site"
    log(f"CV held-out sites: {len(rec) / N_REPEATS:.1f} per repeat, nearest-train quantiles {quantiles(rec['d'])} "
        f"(test {quantiles(d_test)})")

    rec["bin"] = np.digitize(rec["d"], DIST_BINS)
    w_bin = distance_weights(d_test, rec["d"].to_numpy())
    e2 = rec[names].sub(rec["y"], axis=0) ** 2
    w = pd.Series(w_bin[rec["bin"].to_numpy()], index=rec.index)
    weighted = e2.mul(w, axis=0).groupby(rec["repeat"]).sum().div(w.groupby(rec["repeat"]).sum(), axis=0)
    best, table = summarise(weighted)
    table["plain_cv_mse"] = e2.groupby(rec["repeat"]).mean().mean()
    log(f"test-geometry CV MSE weighted to the test distance mix (minus_best paired against {best}); plain alongside:\n"
        f"{table.sort_values('cv_mse').round(4).to_string()}")
    log(f"plain MSE by distance bin:\n{e2.groupby(rec['bin']).mean()[list(dict.fromkeys([best, V1]))].round(3).T.to_string()}")

    gain, gain_se = table.loc[V1, "minus_best"], table.loc[V1, "paired_se"]
    if best != V1 and gain < 2 * gain_se:
        log(f"{best} beats V1 by only {gain:.4f} ± {gain_se:.4f} (< 2 paired SE) -> keep V1")
        best = V1
    preds, m, tef = predict_all(train_all, test_all, specs)
    log(f"full-data Co GP: {({k: (np.round(v, 4).tolist() if isinstance(v, np.ndarray) else round(float(v), 4)) for k, v in m['q'].items()})}")
    lo, hi = train["Co"].min(), train["Co"].max()
    pred = np.clip(preds[best], lo, hi)
    log(f"chosen: {best}, weighted CV {table.loc[best, 'cv_mse']:.4f} (V1 {table.loc[V1, 'cv_mse']:.4f}, V1 minus best "
        f"{gain:.4f} ± {gain_se:.4f}); {int(np.sum(pred != preds[best]))} clipped "
        f"to [{lo}, {hi}]; RMS change vs V1 {np.sqrt(np.mean((pred - preds[V1]) ** 2)):.3f}; correlation with candidates "
        f"{pd.DataFrame(preds).corr()[best].round(3).to_dict()}")
    change = tef.assign(chosen=pred, v1=preds[V1], diff=pred - preds[V1]).iloc[np.argsort(-np.abs(pred - preds[V1]))[:10]]
    log(f"10 test sites changed most vs V1:\n{change[['id', 'Rock', 'Landuse', 'co_lag', 'co_lag_sd', 'lag_dist', 'v1', 'chosen', 'diff']].round(2).to_string(index=False)}")

    sub = pd.DataFrame({"V1": test_all["id"].to_numpy(), "V2": pred})
    assert len(sub) == 100 and (sub["V1"] == np.arange(1, 101)).all() and np.isfinite(sub["V2"]).all()
    sub.to_csv(OUT / "submission.csv", index=False)
    (OUT / "extra").mkdir(exist_ok=True)
    rec.to_csv(OUT / "extra" / "cv_records.csv", index=False)
    pd.DataFrame({"V1": test_all["id"], **preds}).to_csv(OUT / "extra" / "test_preds.csv", index=False)
    (OUT / "extra" / "choice.json").write_text(json.dumps(
        {"chosen": best, "bin_weights": w_bin.tolist(), "specs": specs, "cv": table.round(5).to_dict("index")}, indent=1))
    log(f"SUBMIT: {OUT / 'submission.csv'}")
