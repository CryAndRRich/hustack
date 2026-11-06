import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision

T0 = time.time()
BUDGET_MIN = 45
TRAIN_END = T0 + (BUDGET_MIN - 4) * 60
SEEDS = 3
TAU = [0, 1e-4, 3e-4, 1e-3, 2e-3, 3e-3, 5e-3, 1e-2]
GAIN = np.linspace(0.8, 1.2, 21)
WEIGHTS = "resnet34-b627a593.pth"
CODE = Path("/tmp/chick")
if not CODE.exists():
    CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
from preprocess import N_FOLDS, load, load_dense_chicken, make_folds

OUT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
PARTS, DATA, IMGS = OUT / "parts", CODE / "data.npz", CODE / "imgs.npy"
DC_IMGS, DC_MAPS = CODE / "dc_imgs.npy", CODE / "dc_maps.npy"


def start_workers(jobs, n_gpu, tag, **extra):
    procs, logs = [], []
    n_workers = min(n_gpu, len(jobs))
    for w in range(n_workers):
        env = dict(os.environ, JOBS=",".join(jobs[w::n_workers]), PARTS=str(PARTS), DATA=str(DATA), IMGS=str(IMGS),
                   DC_IMGS=str(DC_IMGS), DC_MAPS=str(DC_MAPS), DEADLINE=str(TRAIN_END),
                   CUDA_VISIBLE_DEVICES=str(w % n_gpu), OMP_NUM_THREADS="4", **{k: str(v) for k, v in extra.items()})
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
        time.sleep(10)
    bad = [w for w, p in enumerate(procs) if p.returncode]
    if bad:
        raise RuntimeError(f"workers {bad} failed; see their log lines above")


def ensure_weights():
    dst = Path(torch.hub.get_dir()) / "checkpoints" / WEIGHTS
    found = [os.path.join(d, WEIGHTS) for d, _, fs in os.walk("/kaggle/input") if WEIGHTS in fs]
    if found:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(found[0], dst)
    else:
        try:
            torchvision.models.resnet34(weights="IMAGENET1K_V1")
        except OSError as err:
            raise RuntimeError(f"no {WEIGHTS}: turn Internet on (Settings > Internet) or attach a dataset "
                               f"holding that file") from err
    print(f"weights: {found[0] if found else 'downloaded'} -> {dst}", flush=True)


def errors(p, t):
    return np.abs(p - t).sum((1, 2)) / t.sum((1, 2))


def post(p, tau, gain):
    return np.where(p < tau, 0.0, p) * gain


def tune(p, t):
    return min(((tau, g) for tau in TAU for g in GAIN), key=lambda q: errors(post(p, *q), t).mean())


def main():
    n_gpu = torch.cuda.device_count()
    assert n_gpu >= 1, "enable the GPU T4 x2 accelerator"
    PARTS.mkdir(parents=True, exist_ok=True)
    ensure_weights()
    tr, te, imgs, Y = load()
    fold = make_folds(imgs[:len(tr)])
    np.save(IMGS, imgs)
    np.savez(DATA, Y=Y, fold=fold)
    dc_imgs, dc_maps = load_dense_chicken()
    np.save(DC_IMGS, dc_imgs)
    np.save(DC_MAPS, dc_maps)
    del imgs, dc_imgs
    print(f"{n_gpu} GPU(s) | train {len(tr)} test {len(te)} | fold sizes {np.bincount(fold).tolist()} "
          f"| Dense-Chicken {len(dc_maps)} frames, {dc_maps.sum():.0f} chickens | data ready {time.time() - T0:.0f}s",
          flush=True)

    wait(*start_workers(["pre:0"], 1, "smoke_pre", PRE_ITERS=10))
    wait(*start_workers(["0:0"], 1, "smoke", ITERS=20, N_PRE=1))
    for f in [*PARTS.glob("*.npy"), *PARTS.glob("*.pt")]:
        f.unlink()
    print(f"smoke ok {time.time() - T0:.0f}s", flush=True)

    n_pre = n_gpu
    wait(*start_workers([f"pre:{s}" for s in range(n_pre)], n_gpu, "pre", PRE_ITERS=os.environ.get("PRE_ITERS", 4000)))
    print(f"pretraining done {time.time() - T0:.0f}s", flush=True)
    jobs = [f"{k}:{s}" for s in range(SEEDS) for k in range(N_FOLDS)]
    wait(*start_workers(jobs, n_gpu, "train", ITERS=os.environ.get("ITERS", 2000), N_PRE=n_pre))
    print(f"training done {time.time() - T0:.0f}s", flush=True)
    merge(te, Y, fold)


def merge(te, Y, fold):
    done = [(k, s) for k in range(N_FOLDS) for s in range(SEEDS) if (PARTS / f"oof_{k}_{s}.npy").exists()]
    missing = sorted(set(range(N_FOLDS)) - {k for k, _ in done})
    assert not missing, f"no finished model for folds {missing}"
    oof, seed0 = np.zeros_like(Y), np.zeros_like(Y)
    for k in range(N_FOLDS):
        seeds = [s for kk, s in done if kk == k]
        oof[fold == k] = np.mean([np.load(PARTS / f"oof_{k}_{s}.npy") for s in seeds], 0)
        seed0[fold == k] = np.load(PARTS / f"oof_{k}_{seeds[0]}.npy")
    test = np.mean([np.load(PARTS / f"test_{k}_{s}.npy") for k, s in done], 0)

    cv = np.zeros_like(Y)
    for k in range(N_FOLDS):
        a, b = fold != k, fold == k
        cv[b] = post(oof[b], *tune(oof[a], Y[a]))
    tau, gain = tune(oof, Y)
    e_raw, e_post = errors(oof, Y), errors(cv, Y)
    folds = " ".join(f"{np.exp(-e_raw[fold == k].mean()):.4f}" for k in range(N_FOLDS))
    print(f"CV score {np.exp(-e_raw.mean()):.4f} (mean e {e_raw.mean():.4f}, {len(done)} models) "
          f"| seed-0 only {np.exp(-errors(seed0, Y).mean()):.4f} | folds {folds}")
    print(f"post-processing cross-fitted CV {np.exp(-e_post.mean()):.4f} | full-data tau {tau} gain {gain:.2f}")
    print(f"count bias: mean pred/true {np.mean(oof.sum((1, 2)) / Y.sum((1, 2))):.3f} "
          f"| count-only error floor {np.mean(np.abs(oof.sum((1, 2)) - Y.sum((1, 2))) / Y.sum((1, 2))):.4f}")
    print("V1 (same folds and seeds, ImageNet init only): CV 0.7502, seed-0 0.7478, "
          "folds 0.7357 0.7505 0.7566 0.7550 0.7533")
    print(f"baselines: flat 30/3600 map {np.exp(-errors(np.full_like(Y, 30 / 3600), Y).mean()):.4f} "
          f"| mean train map {np.exp(-errors(np.broadcast_to(Y.mean(0), Y.shape), Y).mean()):.4f} | LB top 0.736")

    extra = OUT / "extra"
    extra.mkdir(exist_ok=True)
    np.save(extra / "oof_raw.npy", oof)
    np.save(extra / "test_raw.npy", test)
    final = post(test, tau, gain)
    assert final.shape == (len(te), *Y.shape[1:]) and np.isfinite(final).all() and (final >= 0).all()
    sub = pd.DataFrame({"id": te["id"], "density": [" ".join(f"{v:.6g}" for v in m.ravel()) for m in final]})
    sub.to_csv(OUT / "submission.csv", index=False)
    back = pd.read_csv(OUT / "submission.csv")
    assert list(back.columns) == ["id", "density"] and (back["id"] == te["id"]).all()
    assert all(len(s.split()) == Y.shape[1] * Y.shape[2] for s in back["density"])
    shutil.rmtree(PARTS)
    print(f"test predicted counts: mean {final.sum((1, 2)).mean():.1f} range {final.sum((1, 2)).min():.1f}"
          f"-{final.sum((1, 2)).max():.1f}")
    print(f"SUBMIT {OUT / 'submission.csv'} (mean of {len(done)} Dense-Chicken-pretrained resnet34 models, tau {tau} gain {gain:.2f}) "
          f"| total {time.time() - T0:.0f}s")


if __name__ == "__main__":
    main()
