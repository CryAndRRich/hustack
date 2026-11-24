import json
import shutil
import sys
import time
import traceback
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold

from preprocess import load_clean
from models import EXTREME_DATES, FLOORS, REF, arms, fit_predict

WORK = Path("/tmp/work")
OUT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
FULL = {"folds": 5, "repeats": 10}
SMOKE = {"folds": 3, "repeats": 1}
GATE_SD = 2.0
T0 = time.time()


class Tee:
    def __init__(self, path):
        self.f = open(path, "w")
        self.out = sys.stdout

    def write(self, s):
        self.out.write(s)
        self.f.write(s)

    def flush(self):
        self.out.flush()
        self.f.flush()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def cv(train, spec, cfg, seed=2026):
    y = train["rentals"].to_numpy(float)
    keep = (~train["date"].isin(EXTREME_DATES)).to_numpy()
    scores = {f: [] for f in FLOORS}
    scores_x = {f: [] for f in FLOORS}
    oof_sum = np.zeros(len(train))
    rkf = RepeatedKFold(n_splits=cfg["folds"], n_repeats=cfg["repeats"], random_state=seed)
    oof = np.zeros(len(train))
    for i, (a, b) in enumerate(rkf.split(train)):
        oof[b] = fit_predict(spec, train.iloc[a], train.iloc[b])
        if (i + 1) % cfg["folds"] == 0:
            for f in FLOORS:
                p = np.maximum(oof, f)
                scores[f].append(rmse(p, y))
                scores_x[f].append(rmse(p[keep], y[keep]))
            oof_sum += oof
    out = {f: {"rmse": float(np.mean(scores[f])), "rmse_sd": float(np.std(scores[f])),
               "rmse_no_extreme": float(np.mean(scores_x[f])), "per_repeat": scores[f]} for f in FLOORS}
    return out, oof_sum / cfg["repeats"]


def explore(train, test):
    y = train["rentals"]
    log(f"rentals after cleaning: mean {y.mean():.0f} sd {y.std():.0f} min {y.min()} max {y.max()}; "
        f"by year {train.groupby('year')['rentals'].mean().round(0).to_dict()}; constant-4518 RMSE on train {rmse(y, 4518):.0f}")
    log(f"weather train {train['weather'].value_counts().to_dict()} test {test['weather'].value_counts().to_dict()}")
    log("mean rentals by month: " + json.dumps(train.groupby("month")["rentals"].mean().round(0).to_dict()))
    log(f"corr(rentals, temp) {np.corrcoef(y, train['temp_c'])[0, 1]:.3f}, (humidity) {np.corrcoef(y, train['humidity_pct'])[0, 1]:.3f}, "
        f"(wind) {np.corrcoef(y, train['wind_speed'])[0, 1]:.3f}")


def run(cfg, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    train, test, sample, report = load_clean(log)
    explore(train, test)
    results, oofs, specs = {}, {}, {}
    for name, spec in arms().items():
        by_floor, oofs[name] = cv(train, spec, cfg)
        for f, r in by_floor.items():
            key = name if f == 0 else f"{name}_floor{f}"
            results[key], specs[key] = r, (name, f)
        r = by_floor[0]
        log(f"arm {name:20s} CV RMSE {r['rmse']:7.1f} (sd over repeats {r['rmse_sd']:5.1f}; without hurricane days "
            f"{r['rmse_no_extreme']:7.1f}); floors " + " ".join(f"{f}: {by_floor[f]['rmse']:.1f}" for f in FLOORS[1:]))
    ref = np.array(results[REF]["per_repeat"])
    for name, r in results.items():
        d = ref - np.array(r["per_repeat"])
        r["gain_vs_ref"], r["gain_sd"] = float(d.mean()), float(d.std(ddof=1)) if len(d) > 1 else float("inf")
    ranked = sorted(results, key=lambda a: results[a]["rmse"])
    best = ranked[0]
    g = results[best]
    pick = best if best != REF and g["gain_vs_ref"] > GATE_SD * g["gain_sd"] else REF
    log("top-8 by CV RMSE (gain vs V2 reference +- paired sd): " + "; ".join(
        f"{a} {results[a]['rmse']:.1f} ({results[a]['gain_vs_ref']:+.1f} +- {results[a]['gain_sd']:.1f})" for a in ranked[:8]))
    log(f"best {best}; gate gain > {GATE_SD} x paired sd -> pick {pick}")
    extra = outdir / "extra"
    extra.mkdir(exist_ok=True)
    raw = {name: fit_predict(spec, train, test) for name, spec in arms().items()}
    preds = {key: np.maximum(raw[name], f) for key, (name, f) in specs.items()}
    for key, p in preds.items():
        if key in arms() or key == pick:
            pd.DataFrame({"row_id": test["row_id"].to_numpy(), "rentals": p}).sort_values("row_id").to_csv(
                extra / f"{key}.csv", index=False)
    pd.DataFrame({"row_id": train["row_id"], "date": train["date"], "rentals": train["rentals"], **oofs}).to_csv(extra / "oof.csv", index=False)
    base_name, floor = specs[pick]
    res = pd.DataFrame(train["rentals"].to_numpy(float) - np.maximum(oofs[base_name], floor), index=train["date"],
                       columns=["resid"])
    worst = res["resid"].abs().sort_values(ascending=False).head(8)
    log(f"largest OOF residuals of {pick}: " + "; ".join(f"{d} {res.loc[d, 'resid']:+.0f}" for d in worst.index))
    log(f"test predictions of {pick}: mean {preds[pick].mean():.0f} min {preds[pick].min():.0f} max {preds[pick].max():.0f}")
    summary = {"cfg": cfg, "cleaning": report, "cv": results, "pick": pick}
    (outdir / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    return test, sample, preds[pick], pick, results[pick]["rmse"]


def write_submission(test, sample, pred, path):
    sub = pd.DataFrame({"row_id": test["row_id"].to_numpy(), "rentals": pred}).set_index("row_id").loc[sample["row_id"]].reset_index()
    assert sub["row_id"].tolist() == sample["row_id"].tolist() and list(sub.columns) == list(sample.columns)
    assert np.isfinite(sub["rentals"]).all() and (sub["rentals"] >= 0).all()
    sub.to_csv(path, index=False)


def main():
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    sys.stdout = Tee(WORK / "log.txt")
    try:
        log("smoke run")
        run(SMOKE, WORK / "smoke")
        log("smoke ok; full run")
        test, sample, pred, pick, score = run(FULL, WORK / "full")
        write_submission(test, sample, pred, OUT / "submission.csv")
        log(f"SUBMIT: {OUT / 'submission.csv'} (arm {pick}, CV RMSE {score:.1f})")
    except Exception:
        traceback.print_exc(file=sys.stdout)
        raise
    finally:
        sys.stdout.flush()
        with zipfile.ZipFile(OUT / "results.zip", "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(WORK.rglob("*")):
                if f.is_file():
                    z.write(f, f.relative_to(WORK))
        sys.stdout = sys.stdout.out


if __name__ == "__main__":
    main()
