from __future__ import annotations

import csv
import gc
import json
import time
import traceback
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

from common.jobs import claim, worker_setup
from common.training import amp_settings
from common.utils import seed_everything
from common.vision import build_model, expand_input_channels
from cv.augment import GpuAugment
from cv.catalog import neighbor_rows, read_catalog
from cv.config import NUM_CLASSES, Config
from cv.data import ImageStore
from cv.engine import fit, predict

PROB_COLS = [f"p{k}" for k in range(NUM_CLASSES)]
MIN_JOB_S = 300
MIN_JOB_FRACTION = 0.5


def write_probs(path: Path, names: list[str], probs: np.ndarray, labels: list[int] | None = None) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["file_name"] + (["label"] if labels is not None else []) + PROB_COLS)
        for i, name in enumerate(names):
            writer.writerow([name] + ([labels[i]] if labels is not None else []) + [f"{v:.6f}" for v in probs[i]])


def run_job(job: dict, cfg: Config, train_store, test_store, train_pos: list[dict], test_pos: list[dict],
            deadline: float, out: Path, log) -> dict:
    device = train_store.device
    fold = job["fold"]
    seed = cfg.seed + 1000 * fold + job["exp_index"]
    seed_everything(seed)
    model = expand_input_channels(build_model(cfg.arch, NUM_CLASSES, pretrained=cfg.pretrained, log=log), len(cfg.frames))
    model = model.to(device, memory_format=torch.channels_last if cfg.channels_last else torch.contiguous_format)
    train = [r for r in train_pos if r["fold"] != fold]
    val = [r for r in train_pos if r["fold"] == fold]
    nb = np.array(neighbor_rows(train_pos + test_pos, cfg.frames))
    n_train = len(train_pos)
    rows_of = lambda part: nb[[r["row"] for r in part]]
    test_rows = nb[n_train:] - n_train
    have = (nb != np.arange(len(nb))[:, None]).sum(0) if len(cfg.frames) > 1 else None
    log(f"[{job['name']}] {cfg.arch} | image {cfg.image_size} × frames {cfg.frames}"
        + (f" (real frames: {dict(zip(cfg.frames, have.tolist()))}/{len(nb)})" if have is not None else "")
        + f" | train {len(train)} images, val {len(val)} images "
        f"({len({r['storm'] for r in val})} storms) | {cfg.epochs} epochs × batch {cfg.batch_size}, lr {cfg.lr}")

    eval_model, info = fit(cfg, model, train_store, rows_of(train),
                           np.array([r["label"] for r in train]), rows_of(val),
                           np.array([r["label"] for r in val]), len(val) + len(test_pos), deadline, seed, log)
    started = time.time()
    use_amp, amp_dtype = amp_settings(cfg, device)
    aug = GpuAugment(cfg, device)
    batch = cfg.batch_size * 2
    val_probs = predict(eval_model, train_store, rows_of(val), aug, batch, use_amp, amp_dtype, aug.views())
    test_probs = predict(eval_model, test_store, test_rows, aug, batch, use_amp, amp_dtype, aug.views())
    val_labels = [r["label"] for r in val]
    finite = np.isfinite(val_probs).all(1)
    acc = float((val_probs[finite].argmax(1) == np.array(val_labels)[finite]).mean()) if finite.any() else 0.0
    nonfinite = int((~finite).sum() + (~np.isfinite(test_probs).all(1)).sum())

    job_dir = out / "jobs" / job["name"]
    job_dir.mkdir(parents=True, exist_ok=True)
    write_probs(job_dir / "oof.csv", [r["file_name"] for r in val], val_probs, val_labels)
    write_probs(job_dir / "test.csv", [r["file_name"] for r in test_pos], test_probs)
    if cfg.save_models:
        (out / "models").mkdir(exist_ok=True)
        state = {k: (v.half() if v.is_floating_point() else v).cpu() for k, v in eval_model.state_dict().items()}
        torch.save({"state_dict": state, "config": asdict(cfg), "fold": fold}, out / "models" / f"{job['name']}.pt")
    result = {"status": "done", **job, "val_acc_tta": acc, "nonfinite_rows": nonfinite,
              "predict_s": round(time.time() - started, 1), **info}
    (job_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"[{job['name']}] DONE | val acc (EMA, TTA {aug.views()}) = {acc:.4f} | predict {result['predict_s']}s"
        + (f" | WARNING {nonfinite} non-finite rows" if nonfinite else "")
        + (" | stopped early by deadline" if info["stopped_early"] else ""))
    return result


def main() -> None:
    spec, out, log, base, device = worker_setup(Config)

    positives = sorted((r for r in read_catalog(out / "catalog.csv") if r["row"] >= 0), key=lambda r: r["row"])
    train_pos = [r for r in positives if r["split"] == "train"]
    test_pos = positives[len(train_pos):]
    array = np.load(spec["cache"], mmap_mode="r")
    started = time.time()
    train_store = ImageStore(array[:len(train_pos)], device, on_gpu=base.gpu_cache)
    test_store = ImageStore(array[len(train_pos):], device, on_gpu=False)
    gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    log(f"{gpu} | torch {torch.__version__} | train images ({len(train_pos)}) on {train_store.location}, "
        f"test ({len(test_pos)}) on {test_store.location} | load {time.time() - started:.0f}s")

    lock_dir = out / "jobs" / ".claims"
    durations: dict[str, float] = {}
    for job in spec["jobs"]:
        if not claim(lock_dir, job["name"]):
            continue
        remaining = spec["deadline"] - time.time()
        need = MIN_JOB_FRACTION * durations[job["exp"]] if job["exp"] in durations else MIN_JOB_S
        job_dir = out / "jobs" / job["name"]
        if remaining < need:
            log(f"[{job['name']}] SKIP: {remaining / 60:.1f} min left before deadline, need ~{need / 60:.1f}")
            job_dir.mkdir(parents=True, exist_ok=True)
            (job_dir / "result.json").write_text(json.dumps({"status": "skipped", **job}), encoding="utf-8")
            continue
        started = time.time()
        try:
            cfg = replace(base, **spec["experiments"][job["exp"]])
            run_job(job, cfg, train_store, test_store, train_pos, test_pos, spec["deadline"], out, log)
            durations[job["exp"]] = time.time() - started
        except Exception:
            log(f"[{job['name']}] ERROR:\n{traceback.format_exc()}")
            job_dir.mkdir(parents=True, exist_ok=True)
            (job_dir / "result.json").write_text(json.dumps({"status": "failed", **job}), encoding="utf-8")
        finally:
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    log("no jobs left")


if __name__ == "__main__":
    main()
