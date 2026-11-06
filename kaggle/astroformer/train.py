import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score

WORK = Path("/tmp/astro")
sys.path.insert(0, str(WORK))
from net import ARMS, JOBS
from preprocess import find_data, load

DEADLINE, WORKERS_PER_GPU = 25 * 60, 2
OUT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")


def launch(data):
    n_gpu = torch.cuda.device_count()
    assert n_gpu > 0, "GPU accelerator required"
    for f in WORK.glob("[cjt]*_*"):
        f.unlink()
    procs = []
    for g in range(n_gpu):
        for w in range(WORKERS_PER_GPU):
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(g), "OMP_NUM_THREADS": "1"}
            log = open(WORK / f"log_{g}{w}.txt", "w")
            procs.append(subprocess.Popen([sys.executable, str(WORK / "net.py"), str(data), str(WORK)],
                                          env=env, stdout=log, stderr=subprocess.STDOUT))
    print(f"{len(procs)} workers on {n_gpu} GPUs, {len(JOBS)} jobs")
    return procs


def wait(procs, t0):
    while any(p.poll() is None for p in procs) and time.time() - t0 < DEADLINE:
        time.sleep(30)
        print(f"{time.time() - t0:5.0f}s: {len(list(WORK.glob('job_*.npz')))}/{len(JOBS)} jobs done", flush=True)
    for p in procs:
        if p.poll() is None:
            p.kill()
    for f in sorted(WORK.glob("log_*.txt")):
        print(f"--- {f.name}\n{f.read_text()[-4000:]}")


if __name__ == "__main__":
    t0 = time.time()
    data = find_data()
    _, va_y, _ = load(data, "validation")
    _, _, te_ids = load(data, "test")
    sample = pd.read_csv(data / "sample_submission.csv")
    assert (sample["id"].values == te_ids).all()
    wait(launch(data), t0)
    done = {k: dict(np.load(WORK / f"job_{k}.npz")) for k in range(len(JOBS)) if (WORK / f"job_{k}.npz").exists()}
    assert done, "no job finished"
    synth_y = np.r_[np.zeros((va_y == 0).sum()), np.ones((va_y == 0).sum())]
    val, test = {}, {}
    for arm in ARMS:
        for tv in (False, True):
            ks = [k for k in done if JOBS[k][0] == arm and JOBS[k][2] == tv]
            if not ks:
                continue
            test[arm, tv] = np.mean([done[k]["test"] for k in ks], 0)
            if tv:
                print(f"arm {arm} train+val: {len(ks)} seeds (val/synth in-sample, not scored)")
                continue
            val[arm] = np.mean([done[k]["val"] for k in ks], 0)
            synth = np.mean([done[k]["synth"] for k in ks], 0)
            seeds = [round(average_precision_score(va_y, done[k]["val"]), 4) for k in ks]
            print(f"arm {arm} {ARMS[arm]}\n  seeds {seeds} ensemble val AP {average_precision_score(va_y, val[arm]):.4f}"
                  f" synth AP {average_precision_score(synth_y, synth):.4f}")
    rank = lambda d: np.mean([rankdata(v) / len(v) for v in d.values()], 0)
    print(f"rank-mean of train-only arms: val AP {average_precision_score(va_y, rank(val)):.4f}")
    best = max(val, key=lambda a: average_precision_score(va_y, val[a]))
    key = (best, True) if (best, True) in test else (best, False)
    pred = 1 / (1 + np.exp(-test[key]))
    assert len(pred) == 400 and np.isfinite(pred).all() and pred.min() >= 0 and pred.max() <= 1
    np.savez_compressed(OUT / "v3_preds.npz", va_y=va_y,
                        **{f"{n}_{a}_{sd}_{'tv' if tv else 'tr'}": d[n] for k, d in done.items()
                           for (a, sd, tv) in [JOBS[k]] for n in ("val", "test", "synth")})
    path = OUT / "submission.csv"
    pd.DataFrame({"id": te_ids, "transit_probability": pred}).to_csv(path, index=False)
    print(f"chosen arm {best} (best train-only ensemble val AP among {len(val)} arms: optimistic by the pick); submitted {'train+val' if key[1] else 'train-only'} models")
    print(f"done in {time.time() - t0:.0f}s")
    print(f"SUBMIT: {path}")
