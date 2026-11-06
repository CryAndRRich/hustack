import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import ndimage
from sklearn.model_selection import GroupKFold

WORK = Path("/tmp/absorption")
sys.path.insert(0, str(WORK))
from net import COLS, JOBS, NFOLD, arm

DEADLINE, WORKERS_PER_GPU = 27 * 60, 2
OUT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")


def find_data():
    hits = sorted(Path("/kaggle/input").rglob("train_geometry.npz")) if Path("/kaggle/input").exists() else []
    if hits:
        return hits[0].parent
    try:
        return Path(__file__).resolve().parent.parent / "data" / "raw"
    except NameError:
        return Path.cwd() / "data" / "raw"


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
        time.sleep(20)
        print(f"{time.time() - t0:5.0f}s: {len(list(WORK.glob('job_*.npz')))}/{len(JOBS)} jobs done", flush=True)
    for p in procs:
        if p.poll() is None:
            p.kill()
    for f in sorted(WORK.glob("log_*.txt")):
        print(f"--- {f.name}\n{f.read_text()[-3000:]}")


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def report(tr, folds, holes):
    done = {k: dict(np.load(WORK / f"job_{k}.npz")) for k in range(len(JOBS)) if (WORK / f"job_{k}.npz").exists()}
    assert done, "no job finished"
    missing = {seed for k, (fold, seed) in enumerate(JOBS) if fold >= 0 and k not in done}
    seeds = sorted({seed for _, seed in JOBS} - missing)
    Y, lo = tr[COLS].values, tr["t_sub_um"].values < 0.86
    arms = {}
    for g in (0, 1):
        ss = [sd for sd in seeds if arm(sd) == g]
        if ss:
            oof = np.zeros_like(Y)
            for k, (fold, seed) in enumerate(JOBS):
                if fold >= 0 and seed in ss:
                    oof[folds == fold] += done[k]["oof"] / len(ss)
            arms[g] = oof
    if arms:
        arms["blend"] = np.mean(list(arms.values()), 0)
        for name, oof in arms.items():
            fam = pd.Series(np.mean((oof - Y) ** 2, 1)).groupby(tr["family"]).mean() ** 0.5
            print(f"CV arm {name}: rmse {rmse(oof, Y):.5f} | t_sub<0.86 {rmse(oof[lo], Y[lo]):.5f}"
                  f" | hole-free {rmse(oof[holes == 0], Y[holes == 0]):.5f} | family median {fam.median():.5f}")
        oof = arms["blend"]
        print("  blend worst families:", fam.sort_values(ascending=False).head(6).round(4).to_dict())
        for key, g in tr.groupby(["scale_patch", "t_sub_um"]).groups.items():
            print(f"  {key}: n {len(g)} rmse {rmse(oof[g], Y[g]):.5f}")
        np.save(WORK / "oof.npy", oof)
        with zipfile.ZipFile(OUT / "oof.zip", "w") as z:
            z.write(WORK / "oof.npy", "oof.npy")
    print("a", np.round([v["a"] for v in done.values()], 3), "b", np.round([v["b"] for v in done.values()], 3))
    tests = [np.mean([v["test"] for k, v in done.items() if arm(JOBS[k][1]) == g], 0)
             for g in (0, 1) if any(arm(JOBS[k][1]) == g for k in done)]
    print(f"test = mean of {len(tests)} arms over {len(done)} models")
    return np.mean(tests, 0)


def write(test, pred):
    rows = [(f"{tid}_{k + 1}", f"{pred[i, k]:.6f}") for i, tid in enumerate(test["id"]) for k in range(len(COLS))]
    pd.DataFrame(rows, columns=["id", "absorption"]).to_csv(OUT / "submission.csv", index=False)
    print(f"SUBMIT submission.csv: {len(rows)} rows")


if __name__ == "__main__":
    t0 = time.time()
    data = find_data()
    tr, te = pd.read_csv(data / "train.csv"), pd.read_csv(data / "test.csv")
    folds = np.zeros(len(tr), dtype=int)
    for f, (_, val) in enumerate(GroupKFold(NFOLD).split(tr, groups=tr["family"])):
        folds[val] = f
    np.save(WORK / "folds.npy", folds)
    wait(launch(data), t0)
    holes = np.array([ndimage.label(p <= 0)[1] - 1 for p in np.load(data / "train_geometry.npz")["phi"]])
    pred = report(tr, folds, holes)
    assert pred.shape == (len(te), len(COLS)) and np.isfinite(pred).all()
    write(te, pred)
    print(f"total {time.time() - t0:.0f}s")
