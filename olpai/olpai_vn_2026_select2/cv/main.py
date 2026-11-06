from __future__ import annotations

import json
import os
import shutil
import zipfile
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from common.distributed import RESULT_FILE, cli, run_or_launch, run_ranks
from common.training import amp_settings
from common.utils import Logger, seed_everything
from common.vision import build_model
from cv.config import Config, find_data_root, resolve_output_dir
from cv.data import CachedImages, RandomJPEG, build_cache, list_images, load_train, open_cache
from cv.engine import fit, make_transforms, predict
from cv.metrics import best_threshold, macro_f1, safe_auc, validate_submission


def prepare(cfg: Config) -> Config:
    data_root = find_data_root(cfg.data_root)
    output_dir = resolve_output_dir(cfg.output_dir)
    cfg, output_dir = cfg.for_run(output_dir)
    cache_dir = cfg.cache_dir_or("tacvu1_cache")
    return replace(cfg, data_root=str(data_root), output_dir=str(output_dir), cache_dir=str(cache_dir))


def run(cfg: Config | None = None) -> dict:
    cfg = prepare(cfg or Config())
    output_dir = Path(cfg.output_dir)
    log = Logger(output_dir / "train.log")

    data_root, cache_dir = Path(cfg.data_root), Path(cfg.cache_dir)
    if not cfg.checkpoint:
        build_cache(data_root / "train" / "images", load_train(data_root).file_name.tolist(), cache_dir, log=log)
    build_cache(data_root / cfg.split / "images", list_images(data_root, cfg.split), cache_dir, log=log)

    return run_or_launch(cfg, "cv.main", execute, log)


def execute(cfg: Config) -> dict | None:
    return run_ranks(_execute, cfg)


def _execute(cfg: Config, ctx) -> dict | None:
    output_dir = Path(cfg.output_dir)
    log = Logger(output_dir / "train.log", enabled=ctx.is_main)
    if cfg.num_workers is None:
        cfg = replace(cfg, num_workers=min(4, max(1, (os.cpu_count() or 2) // ctx.world_size)))
    seed_everything(cfg.seed, cfg.deterministic)

    device = ctx.device
    gpu = f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""
    log(f"[config] {json.dumps(asdict(cfg), ensure_ascii=False)}")
    log(f"[config] device={device}{gpu} × {ctx.world_size} | torch {torch.__version__}")

    data_root, cache_dir = Path(cfg.data_root), Path(cfg.cache_dir)
    test_names = list_images(data_root, cfg.split)
    _, test_array = open_cache(cache_dir, data_root / cfg.split / "images", test_names)
    test_positions = np.arange(len(test_names))
    if cfg.debug:
        test_positions = test_positions[:16]
        log("[debug] running on a subset — RESULTS NOT FOR SUBMISSION")

    if cfg.checkpoint:
        ckpt = torch.load(cfg.checkpoint, map_location="cpu", weights_only=False)
        cfg = replace(cfg, arch=ckpt["arch"], image_size=ckpt["image_size"])
        memory_format = torch.channels_last if cfg.channels_last else torch.contiguous_format
        models = []
        for key in ("state_dict", "refit_state_dict"):
            if ckpt.get(key) is not None:
                m = build_model(cfg.arch, pretrained=False, log=log)
                m.load_state_dict(ckpt[key])
                models.append(m.to(device, memory_format=memory_format))
        threshold, val = ckpt["threshold"], ckpt.get("val", {})
        log(f"[checkpoint] {cfg.checkpoint} | arch={cfg.arch} | image={cfg.image_size}px | threshold={threshold:.2f} "
            f"| {len(models)} model")
        val_names, val_labels, val_probs = [], [], np.empty(0)
    else:
        models, threshold, val, (val_names, val_labels, val_probs) = _train(cfg, ctx, log)

    use_amp, amp_dtype = amp_settings(cfg, device)
    _, eval_tf = make_transforms(cfg, device)
    probs_each = [predict(m, eval_tf, test_array, test_positions, cfg, ctx, use_amp, amp_dtype, tta=cfg.tta_hflip)
                  for m in models]
    test_probs = np.mean(probs_each, axis=0)
    if not ctx.is_main:
        return None
    if len(probs_each) > 1:
        agree = float(((probs_each[0] >= threshold) == (probs_each[1] >= threshold)).mean())
        log(f"[refit] val model vs refit model agree on {agree:.1%} of test labels | averaging both")

    names = [test_names[i] for i in test_positions]
    submission = pd.DataFrame({"file_name": names, "category_id": (test_probs >= threshold).astype(int)})
    validate_submission(submission, names)
    csv_path, zip_path = output_dir / "submission.csv", output_dir / "submission.zip"
    submission.to_csv(csv_path, index=False)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(csv_path, arcname="submission.csv")
    if cfg.split == "private_test":
        shutil.copyfile(zip_path, output_dir / "private_submission.zip")
    log(f"[submit] {zip_path} | threshold={threshold:.2f} "
        f"| label counts: {submission.category_id.value_counts().to_dict()}")

    pd.DataFrame({"file_name": names, "prob_fake": test_probs}) \
        .to_csv(output_dir / f"{cfg.split}_probs.csv", index=False)
    if not cfg.checkpoint:
        torch.save({
            "arch": cfg.arch,
            "image_size": cfg.image_size,
            "num_classes": 2,
            "threshold": threshold,
            "state_dict": {k: v.cpu() for k, v in models[0].state_dict().items()},
            "refit_state_dict": ({k: v.cpu() for k, v in models[1].state_dict().items()}
                                 if len(models) > 1 else None),
            "config": asdict(cfg),
            "val": val,
        }, output_dir / "best_model.pt")
        pd.DataFrame({"file_name": val_names, "label": val_labels, "prob_fake": val_probs}) \
            .to_csv(output_dir / "val_preds.csv", index=False)

    result = {"val": val, "threshold": threshold, "submission_csv": str(csv_path),
              "submission_zip": str(zip_path), "output_dir": str(output_dir), "config": asdict(cfg)}
    (output_dir / RESULT_FILE).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _train(cfg: Config, ctx, log):
    data_root, cache_dir = Path(cfg.data_root), Path(cfg.cache_dir)
    train_df = load_train(data_root)
    train_cache, train_array = open_cache(cache_dir, data_root / "train" / "images", train_df.file_name.tolist())
    labels = train_df.label.to_numpy()

    positions = np.arange(len(train_df))
    if cfg.debug:
        positions = train_df.groupby("label").sample(n=16, random_state=cfg.seed).index.to_numpy()
    train_pos, val_pos = train_test_split(positions, test_size=cfg.val_size,
                                          stratify=labels[positions], random_state=cfg.seed)
    val_pos = np.sort(val_pos)
    log(f"[data] train={len(train_pos)} {np.bincount(labels[train_pos]).tolist()} "
        f"| val={len(val_pos)} {np.bincount(labels[val_pos]).tolist()} | {cfg.split} predicted later")

    model = _new_model(cfg, ctx, log)
    train_ds = CachedImages(train_cache, train_pos, labels, cpu_transform=RandomJPEG(p=cfg.jpeg_p))
    best = fit(cfg, ctx, model, train_ds, train_array, val_pos, labels[val_pos], log)

    use_amp, amp_dtype = amp_settings(cfg, ctx.device)
    _, eval_tf = make_transforms(cfg, ctx.device)
    val_probs = predict(model, eval_tf, train_array, val_pos, cfg, ctx, use_amp, amp_dtype, tta=cfg.tta_hflip)
    val_labels = labels[val_pos]
    f1_default = macro_f1(val_labels, (val_probs >= 0.5).astype(int))
    tuned, f1_tuned = best_threshold(val_labels, val_probs)
    threshold = tuned if cfg.tune_threshold else 0.5
    val = {
        "best_epoch": best["epoch"],
        "f1@0.5": f1_default,
        "f1@tuned": f1_tuned,
        "tuned_threshold": tuned,
        "threshold_used": threshold,
        "auc": safe_auc(val_labels, val_probs),
        "loss_no_tta": best["loss"],
    }
    log(f"[val] best epoch={best['epoch']} | Macro F1 @0.5={f1_default:.4f} | @{tuned:.2f}={f1_tuned:.4f} "
        f"| AUC={val['auc']:.4f} | using threshold {threshold:.2f}" + (" (TTA hflip)" if cfg.tta_hflip else ""))
    val_names = train_df.file_name.to_numpy()[val_pos].tolist()
    models = [model]

    if cfg.refit:
        log(f"[refit] retraining on {len(positions)} images up to epoch {best['epoch']}")
        refit_model = _new_model(cfg, ctx, log)
        all_ds = CachedImages(train_cache, positions, labels, cpu_transform=RandomJPEG(p=cfg.jpeg_p))
        fit(cfg, ctx, refit_model, all_ds, None, None, None, log, stop_epoch=best["epoch"])
        models.append(refit_model)
    return models, threshold, val, (val_names, val_labels, val_probs)


def _new_model(cfg: Config, ctx, log):
    if not ctx.is_main:
        ctx.barrier()
    model = build_model(cfg.arch, pretrained=cfg.pretrained, pretrained_dir=cfg.pretrained_dir, log=log)
    if ctx.is_main:
        ctx.barrier()
    model = model.to(ctx.device, memory_format=torch.channels_last if cfg.channels_last else torch.contiguous_format)
    torch.manual_seed(cfg.seed + ctx.rank)
    return model


if __name__ == "__main__":
    cli(Config, run, execute)
