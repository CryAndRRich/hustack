import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from preprocess import load, offset
from train import N_FOLDS

SEEDS = 3
WORKERS_PER_GPU = 2
OUT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
PARTS = OUT / "parts"
try:
    HERE = Path(__file__).resolve().parent
except NameError:
    HERE = Path.cwd()


def launch():
    n_gpu = torch.cuda.device_count()
    jobs = [f"{k}:{s}" for s in range(SEEDS) for k in range(N_FOLDS)]
    n = max(n_gpu * WORKERS_PER_GPU, 1)
    print(f"{n_gpu} GPU(s), {n} workers, {len(jobs)} jobs", flush=True)
    procs, logs, seen = [], [], []
    for w in range(n):
        env = dict(os.environ, JOBS=",".join(jobs[w::n]), PARTS=str(PARTS), OMP_NUM_THREADS="1")
        if n_gpu:
            env["CUDA_VISIBLE_DEVICES"] = str(w % n_gpu)
        logs.append(PARTS / f"worker{w}.log")
        procs.append(subprocess.Popen([sys.executable, "-u", str(HERE / "train.py")], env=env, cwd=HERE,
                                      stdout=open(logs[w], "w"), stderr=subprocess.STDOUT))
        seen.append(0)
    while True:
        done = all(p.poll() is not None for p in procs)
        for w, log in enumerate(logs):
            text = log.read_text()
            for line in text[seen[w]:].splitlines():
                print(f"w{w} {line}", flush=True)
            seen[w] = len(text)
        if done:
            break
        time.sleep(15)
    bad = [w for w, p in enumerate(procs) if p.returncode]
    if bad:
        raise RuntimeError(f"workers {bad} failed; see their log lines above")


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def merge():
    tr, _, ytr, te, _ = load()
    oof = {(k, s): np.load(PARTS / f"oof_{k}_{s}.npy") for k in range(N_FOLDS) for s in range(SEEDS)}
    test = np.mean([np.load(PARTS / f"test_{k}_{s}.npy") for k in range(N_FOLDS) for s in range(SEEDS)], axis=0)
    per_seed = [np.nanmean([oof[k, s] for k in range(N_FOLDS)], axis=0) for s in range(SEEDS)]
    ens = np.nanmean([oof[key] for key in oof], axis=0)
    print("CV rmse per seed:", [round(rmse(p, ytr), 4) for p in per_seed], f"| {SEEDS}-seed ensemble {rmse(ens, ytr):.4f}")
    for k in range(N_FOLDS):
        m = ~np.isnan(oof[k, 0])
        print(f"  fold {k}: ensemble {rmse(ens[m], ytr[m]):.4f}")
    (OUT / "extra").mkdir(exist_ok=True)
    np.save(OUT / "extra" / "oof.npy", ens)
    pd.DataFrame({"row_id": te["row_id"], "temp_hotspot_c": test + offset(te)}).to_csv(
        OUT / "submission.csv", index=False)
    shutil.rmtree(PARTS)
    print(f"SUBMIT {OUT / 'submission.csv'} ({len(oof)} models)")


if __name__ == "__main__":
    t = time.time()
    PARTS.mkdir(parents=True, exist_ok=True)
    launch()
    merge()
    print(f"total {time.time() - t:.0f}s")
