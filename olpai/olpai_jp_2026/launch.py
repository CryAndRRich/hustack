import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.optimize import nnls

T0 = time.time()
BUDGET_MIN = 60
TRAIN_END = T0 + (BUDGET_MIN - 7) * 60
TEACHERS = ["gru", "tcn", "tf"]
STUDENTS = [f"s_{a}" for a in TEACHERS]
SEEDS = 2
WORKERS_PER_GPU = 2
CODE = Path("/tmp/joai")
if not CODE.exists():
    CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
from preprocess import N_FOLDS, arrays, load

OUT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
PARTS, DATA = OUT / "parts", CODE / "data.npz"


def start_workers(jobs, n_gpu, n_workers, epochs, tag):
    procs, logs = [], []
    for w in range(n_workers):
        env = dict(os.environ, JOBS=",".join(jobs[w::n_workers]), PARTS=str(PARTS), DATA=str(DATA),
                   DEADLINE=str(TRAIN_END), EPOCHS=str(epochs), CUDA_VISIBLE_DEVICES=str(w % n_gpu),
                   OMP_NUM_THREADS="1")
        logs.append(PARTS / f"{tag}{w}.log")
        procs.append(subprocess.Popen([sys.executable, "-u", str(CODE / "train.py")], env=env, cwd=CODE,
                                      stdout=open(logs[-1], "w"), stderr=subprocess.STDOUT))
    return procs, logs


def wait(procs, logs):
    seen = [0] * len(logs)
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


def mse(p, y):
    return float(np.mean((p - y) ** 2))


def grid(arms, folds, seeds):
    return [f"{a}:{k}:{s}" for s in seeds for a in arms for k in folds]


def teach(z, folds):
    hi = float(z["ytr"].max())
    for k in folds:
        b = z["fold"] == k
        files = [f for a in TEACHERS for s in range(SEEDS) if (f := PARTS / f"oof_{a}_{k}_{s}.npy").exists()]
        assert files, f"no teacher models for fold {k}"
        oof = np.clip(np.mean([np.load(f) for f in files], 0), 0, hi)
        test = np.clip(np.mean([np.load(f.with_name(f.name.replace("oof_", "test_"))) for f in files], 0), 0, hi)
        mask = np.arange(oof.shape[1])[None] < z["ntr"][b][:, None]
        score = float(((oof[b] - z["ytr"][b]) ** 2)[mask].mean())
        np.savez(PARTS / f"teach_{k}.npz", oof=oof.astype(np.float32), test=test.astype(np.float32))
        print(f"teacher fold {k}: mean of {len(files)} models, valid mse {score:.4f}", flush=True)


def main():
    n_gpu = torch.cuda.device_count()
    assert n_gpu >= 1, "enable the GPU T4 x2 accelerator"
    PARTS.mkdir(parents=True, exist_ok=True)
    tr, te, brain = load()
    z = arrays(tr, te, brain)
    np.savez(DATA, **z)
    print(f"{n_gpu} GPU(s) | train {len(tr)} rows / {len(z['ntr'])} windows | test {len(te)} / {len(z['nte'])} "
          f"| {z['Xtr'].shape[2]} features | {z['n_md']} mouse-days "
          f"| data ready {time.time() - T0:.0f}s", flush=True)

    wait(*start_workers(grid(TEACHERS, [0], [0]), 1, 1, 1, "smoke"))
    teach(z, [0])
    wait(*start_workers(grid(STUDENTS, [0], [0]), 1, 1, 1, "smoke_s"))
    for f in [*PARTS.glob("*.npy"), *PARTS.glob("*.npz")]:
        f.unlink()
    print(f"smoke ok {time.time() - T0:.0f}s", flush=True)

    folds, n_workers, epochs = range(N_FOLDS), n_gpu * WORKERS_PER_GPU, int(os.environ.get("EPOCHS", 40))
    wait(*start_workers(grid(TEACHERS, folds, range(SEEDS)), n_gpu, n_workers, epochs, "teacher"))
    print(f"teachers done {time.time() - T0:.0f}s", flush=True)
    teach(z, folds)
    wait(*start_workers(grid(STUDENTS, folds, range(SEEDS)), n_gpu, n_workers, epochs, "student"))
    print(f"students done {time.time() - T0:.0f}s", flush=True)
    merge(tr, te, z)


def merge(tr, te, z):
    y = tr["lever"].to_numpy()
    wtr, ttr, wte, tte = z["wtr"], z["ttr"], z["wte"], z["tte"]
    lo, hi = 0.0, float(y.max())
    row_fold = z["fold"][wtr]
    extra = OUT / "extra"
    extra.mkdir(exist_ok=True)
    order = np.argsort(tr["row"].to_numpy())

    def write(path, pred_sorted):
        out = np.empty(len(te), np.float32)
        out[te["row"].to_numpy()] = pred_sorted
        ids = np.empty(len(te), object)
        ids[te["row"].to_numpy()] = te["id"].to_numpy()
        pd.DataFrame({"id": ids, "lever": out}).to_csv(path, index=False)

    oof, test, seed0 = {}, {}, {}
    for arm in TEACHERS + STUDENTS:
        done = [(k, s) for k in range(N_FOLDS) for s in range(SEEDS) if (PARTS / f"oof_{arm}_{k}_{s}.npy").exists()]
        if {k for k, _ in done} != set(range(N_FOLDS)):
            print(f"{arm}: incomplete ({len(done)} models, folds {sorted({k for k, _ in done})}); dropped")
            continue
        o = np.nanmean([np.load(PARTS / f"oof_{arm}_{k}_{s}.npy") for k, s in done], axis=0)
        oof[arm] = np.clip(o[wtr, ttr], lo, hi)
        test[arm] = np.clip(np.mean([np.load(PARTS / f"test_{arm}_{k}_{s}.npy") for k, s in done], axis=0)[wte, tte], lo, hi)
        if all((k, 0) in done for k in range(N_FOLDS)):
            o0 = np.nanmean([np.load(PARTS / f"oof_{arm}_{k}_0.npy") for k in range(N_FOLDS)], axis=0)
            seed0[arm] = mse(np.clip(o0[wtr, ttr], lo, hi), y)
        folds = " ".join(f"{mse(oof[arm][row_fold == k], y[row_fold == k]):.4f}" for k in range(N_FOLDS))
        print(f"{arm:6s} CV {mse(oof[arm], y):.4f} ({len(done)} models) | seed-0 only {seed0.get(arm, float('nan')):.4f} "
              f"| folds {folds}")
        np.save(extra / f"oof_{arm}.npy", oof[arm][order])
        write(extra / f"submission_{arm}.csv", test[arm])
    print(f"predict-mean baseline {np.var(y):.4f}; V2 (same recipe as the teachers): gru 0.7802, tcn 0.7904, tf 0.8610, "
          f"stack 0.7604")
    for a in TEACHERS:
        if a in seed0 and f"s_{a}" in seed0:
            print(f"student vs teacher seed-0 {a}: {seed0[a]:.4f} -> {seed0[f's_{a}']:.4f} "
                  f"(delta {seed0[f's_{a}'] - seed0[a]:+.4f})")

    def stack(arms, name):
        P, Pte = np.column_stack([oof[a] for a in arms]), np.column_stack([test[a] for a in arms])
        cv = np.zeros(len(y))
        for k in range(N_FOLDS):
            a, b = row_fold != k, row_fold == k
            cv[b] = P[b] @ nnls(P[a], y[a])[0]
        wts = nnls(P, y)[0]
        folds = " ".join(f"{mse(cv[row_fold == k], y[row_fold == k]):.4f}" for k in range(N_FOLDS))
        print(f"stack {name} {arms}: cross-fitted CV {mse(cv, y):.4f} | equal mean {mse(P.mean(1), y):.4f} "
              f"| weights {dict(zip(arms, wts.round(3)))} | folds {folds}")
        np.save(extra / f"oof_stack_{name}.npy", cv[order])
        write(extra / f"submission_stack_{name}.csv", np.clip(Pte @ wts, lo, hi))
        return np.clip(Pte @ wts, lo, hi)

    for name, group in [("teachers", TEACHERS), ("students", STUDENTS)]:
        if all(a in oof for a in group):
            stack(group, name)
    arms = [a for a in TEACHERS + STUDENTS if a in oof]
    write(OUT / "submission.csv", stack(arms, "all"))
    sub = pd.read_csv(OUT / "submission.csv")
    assert len(sub) == len(te) and sub["lever"].notna().all() and list(sub.columns) == ["id", "lever"]
    shutil.rmtree(PARTS)
    print(f"SUBMIT {OUT / 'submission.csv'} (NNLS stack of {arms}) | total {time.time() - T0:.0f}s")


if __name__ == "__main__":
    main()
