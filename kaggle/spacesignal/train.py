import json
import shutil
import sys
import time
import traceback
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from preprocess import load
from models import fit_signal

WORK = Path("/tmp/work")
OUT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
RMS_OK = 1e-6
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


def q(x, qs=(0, 0.1, 0.5, 0.9, 1)):
    return " ".join(f"{v:.4g}" for v in np.quantile(x, qs))


def explore(data):
    t, Y = data["t"], data["Y"]
    dt = np.diff(t)
    e = {"dt_min": float(dt.min()), "dt_max": float(dt.max()), "linspace": bool(np.allclose(t, np.linspace(t[0], t[-1], len(t)))),
         "d4_median_q": np.quantile(np.median(np.abs(np.diff(Y, 4, axis=0)), 0), [0, 0.5, 1]).tolist(),
         "absmax_q": np.quantile(np.abs(Y).max(0), [0, 0.5, 1]).tolist()}
    log(f"time grid dt {e['dt_min']:.9f}-{e['dt_max']:.9f} linspace {e['linspace']}; |y| max per signal q0/50/100 "
        f"{q(np.abs(Y).max(0), (0, 0.5, 1))}; median |4th diff| q0/50/100 {q(np.median(np.abs(np.diff(Y, 4, axis=0)), 0), (0, 0.5, 1))}")
    return e


def fit_all(data, idx):
    t, Y = data["t"], data["Y"]
    rows = []
    for i in idx:
        r = fit_signal(t, Y[:, i], seed=1000 + int(i))
        r["ID"] = int(data["ids"][i])
        rows.append(r)
        if r["rms"] > RMS_OK or r["fallback"]:
            log(f"signal {i}: rms {r['rms']:.3g} fallback {r['fallback']} zeros {r['n_zeros']}/{r['n_zeros_resolved']}")
    return pd.DataFrame(rows)


def submission(tab, convention):
    a, b, g = tab["alpha"].to_numpy(), tab["beta"].to_numpy(), tab["gamma"].to_numpy()
    if convention == "beta_pos":
        flip = b < 0
        a, b = np.where(flip, -a, a), np.where(flip, -b, b)
    sub = pd.DataFrame({"ID": tab["ID"].to_numpy(), "alpha": a, "beta": b, "gamma": g})
    return sub


def write_submission(sub, path, n):
    assert len(sub) == n and sub["ID"].tolist() == list(range(n)), "IDs must be 0..n-1 in order"
    assert list(sub.columns) == ["ID", "alpha", "beta", "gamma"]
    assert np.isfinite(sub[["alpha", "beta", "gamma"]].to_numpy()).all(), "non-finite parameter"
    sub.to_csv(path, index=False)


def run(idx, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    data = load()
    summary = {"explore": explore(data)}
    log(f"fitting {len(idx)} signals")
    tab = fit_all(data, idx)
    rel = tab["rms"] / tab["y_absmax"]
    n_bad = int((tab["rms"] > RMS_OK).sum())
    summary.update({"rms_q": np.quantile(tab["rms"], [0, 0.5, 0.9, 1]).tolist(), "n_rms_gt_1e-6": n_bad,
                    "n_fallback": int(tab["fallback"].sum())})
    log(f"rms q0/50/90/100 {q(tab['rms'], (0, 0.5, 0.9, 1))}; rms / max|y| q50/100 {q(rel, (0.5, 1))}; "
        f"signals with rms > {RMS_OK:g}: {n_bad}; fallback used: {int(tab['fallback'].sum())}")
    if "alpha_init" in tab:
        err = (tab["alpha_init"] - tab["alpha"]).abs() / tab["alpha"]
        log(f"init vs final: relative alpha error q50/90/100 {q(err.dropna(), (0.5, 0.9, 1))}; "
            f"t0 error q50/90/100 {q((tab['t0_init'] - tab['t0']).abs().dropna(), (0.5, 0.9, 1))}")
    for c in ("alpha", "beta", "gamma", "t0"):
        log(f"  {c} (alpha > 0 convention) q0/10/50/90/100 {q(tab[c])}; n < 0: {int((tab[c] < 0).sum())}")
    log(f"  under beta > 0: alpha < 0 for {int((tab['beta'] < 0).sum())} signals")
    tab.to_csv(outdir / "fits.csv", index=False)
    extra = outdir / "extra"
    extra.mkdir(exist_ok=True)
    subs = {c: submission(tab, c) for c in ("alpha_pos", "beta_pos")}
    for c, s in subs.items():
        s.to_csv(extra / f"{c}.csv", index=False)
    mae_between = float(np.abs(subs["alpha_pos"][["alpha", "beta", "gamma"]].to_numpy()
                               - subs["beta_pos"][["alpha", "beta", "gamma"]].to_numpy()).mean())
    summary["mae_alpha_pos_vs_beta_pos"] = mae_between
    log(f"MAE between the alpha_pos and beta_pos files: {mae_between:.5f}")
    (outdir / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    return subs["alpha_pos"], len(data["ids"])


def main():
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    sys.stdout = Tee(WORK / "log.txt")
    try:
        log("smoke run (3 signals)")
        run([0, 5, 40], WORK / "smoke")
        log("smoke ok; full run")
        sub, n = run(list(range(100)), WORK / "full")
        write_submission(sub, OUT / "submission.csv", n)
        log(f"SUBMIT: {OUT / 'submission.csv'} (convention alpha > 0)")
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
