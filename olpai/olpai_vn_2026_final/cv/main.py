from __future__ import annotations

import json
import os
import zipfile
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common.distributed import RESULT_FILE, cli, run_or_launch, run_ranks
from common.paths import unpack_split
from common.training import amp_settings
from common.utils import Logger, seed_everything
from common.vision import build_model
from cv.augment import make_transforms
from cv.config import CLASSES, Config, default_pretrained_dir, find_data_root, resolve_output_dir
from cv.data import CachedImages, build_cache, clean_train, list_images, open_cache
from cv.engine import fit, predict
from cv.grouping import build_groups, group_split
from cv.metrics import NUM_CLASSES, confusion_table, summarize, validate_submission

TRAIN_TABLE = "train_clean.csv"
PROB_COLS = [f"p_{c}" for c in CLASSES]


def prepare(cfg: Config) -> Config:
    data_root = find_data_root(cfg.data_root)
    output_dir = resolve_output_dir(cfg.output_dir)
    cfg, output_dir = cfg.for_run(output_dir)
    cache_dir = cfg.cache_dir_or("tacvu2_cache")
    test_root = cfg.test_root or unpack_split(data_root, cfg.split, cfg.zip_password, cache_dir / "unpacked")
    return replace(cfg, data_root=str(data_root), output_dir=str(output_dir), cache_dir=str(cache_dir),
                   test_root=str(test_root), pretrained_dir=default_pretrained_dir(cfg.pretrained_dir))


def prepare_train(cfg: Config, log) -> pd.DataFrame:
    data_root, cache_dir = Path(cfg.data_root), Path(cfg.cache_dir)
    train_df = clean_train(data_root, cfg.min_copies, log)
    if cfg.debug:
        train_df = train_df.groupby("label").head(20).reset_index(drop=True)
    names = train_df.file_name.tolist()
    image_dir = data_root / "train" / "images"
    build_cache(image_dir, names, cache_dir, log=log)
    _, array = open_cache(cache_dir, image_dir, names)
    train_df["group"] = build_groups(cfg, names, train_df.label.to_numpy(), array, log)
    train_df.to_csv(Path(cfg.output_dir) / TRAIN_TABLE, index=False)
    return train_df


def run(cfg: Config | None = None) -> dict:
    cfg = prepare(cfg or Config())
    output_dir = Path(cfg.output_dir)
    log = Logger(output_dir / "train.log")

    test_root, cache_dir = Path(cfg.test_root), Path(cfg.cache_dir)
    if not cfg.checkpoint:
        prepare_train(cfg, log)
    build_cache(test_root / cfg.split / "images", list_images(test_root, cfg.split), cache_dir, log=log)

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

    test_root, cache_dir = Path(cfg.test_root), Path(cfg.cache_dir)
    test_names = list_images(test_root, cfg.split)
    _, test_array = open_cache(cache_dir, test_root / cfg.split / "images", test_names)
    log(f"[test] {cfg.split}: {len(test_names)} images ({test_root / cfg.split / 'images'})")
    test_positions = np.arange(len(test_names))
    if cfg.debug:
        test_positions = test_positions[:16]
        log("[debug] running on a subset — RESULTS NOT FOR SUBMISSION")

    if cfg.checkpoint:
        ckpt = torch.load(cfg.checkpoint, map_location="cpu", weights_only=False)
        cfg = replace(cfg, arch=ckpt["arch"], image_size=ckpt["image_size"])
        model = build_model(cfg.arch, num_classes=NUM_CLASSES, pretrained=False, log=log)
        model.load_state_dict(ckpt["state_dict"])
        model = model.to(device, memory_format=torch.channels_last if cfg.channels_last else torch.contiguous_format)
        val, val_frame = ckpt.get("val", {}), None
        log(f"[checkpoint] {cfg.checkpoint} | arch={cfg.arch} | image={cfg.input_hw}")
    else:
        model, val, val_frame = _train(cfg, ctx, log)

    use_amp, amp_dtype = amp_settings(cfg, device)
    _, eval_tf = make_transforms(cfg, device)
    test_probs = predict(model, eval_tf, test_array, test_positions, cfg, ctx, use_amp, amp_dtype,
                         tta=cfg.tta_hflip)
    if not ctx.is_main:
        return None

    names = [test_names[i] for i in test_positions]
    csv_path, zip_path = write_submission(names, test_probs, output_dir, log)
    pd.DataFrame(test_probs, columns=PROB_COLS).assign(file_name=names)[["file_name", *PROB_COLS]] \
        .to_csv(output_dir / f"{cfg.split}_probs.csv", index=False)
    if not cfg.checkpoint:
        torch.save({
            "arch": cfg.arch,
            "image_size": cfg.image_size,
            "num_classes": NUM_CLASSES,
            "classes": CLASSES,
            "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "config": asdict(cfg),
            "val": val,
        }, output_dir / "best_model.pt")
        val_frame.to_csv(output_dir / "val_preds.csv", index=False)

    result = {"val": val, "submission_csv": str(csv_path), "submission_zip": str(zip_path),
              "output_dir": str(output_dir), "config": asdict(cfg)}
    (output_dir / RESULT_FILE).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def write_submission(names: list[str], probs: np.ndarray, output_dir: Path, log=print):
    submission = pd.DataFrame({"file_name": names, "category_id": probs.argmax(1) + 1})
    validate_submission(submission, names)
    csv_path, zip_path = output_dir / "submission.csv", output_dir / "submission.zip"
    submission.to_csv(csv_path, index=False)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(csv_path, arcname="submission.csv")
    counts = submission.category_id.value_counts().reindex(range(1, NUM_CLASSES + 1), fill_value=0)
    log(f"[submit] {zip_path} | {len(submission)} images | prediction distribution "
        f"{dict(zip(CLASSES, counts.tolist()))}")
    return csv_path, zip_path


def log_validation(tag: str, labels: np.ndarray, probs: np.ndarray, log) -> dict:
    metrics = summarize(labels, probs)
    log(f"[{tag}] Macro F1={metrics['f1']:.4f} | acc={metrics['acc']:.4f} | loss={metrics['loss']:.4f}"
        f"\n  per-class F1: {metrics['per_class_f1']}"
        f"\n  confusion matrix (rows = true, cols = predicted):\n"
        + confusion_table(labels, probs.argmax(1)).to_string())
    return metrics


def _train(cfg: Config, ctx, log):
    data_root, cache_dir = Path(cfg.data_root), Path(cfg.cache_dir)
    train_df = pd.read_csv(Path(cfg.output_dir) / TRAIN_TABLE)
    train_cache, train_array = open_cache(cache_dir, data_root / "train" / "images", train_df.file_name.tolist())
    labels = train_df.label.to_numpy()
    groups = train_df.group.to_numpy()

    train_pos, val_pos = group_split(labels, groups, cfg.val_size, cfg.seed)
    shared = len(np.intersect1d(groups[train_pos], groups[val_pos]))
    log(f"[data] train={len(train_pos)} {np.bincount(labels[train_pos], minlength=NUM_CLASSES).tolist()} "
        f"| val={len(val_pos)} {np.bincount(labels[val_pos], minlength=NUM_CLASSES).tolist()} "
        f"| split={cfg.split_mode} ({len(np.unique(groups[val_pos]))} val groups, groups shared by both sides: {shared}) "
        f"| {cfg.split} predicted later")

    model = _new_model(cfg, ctx, log)
    train_ds = CachedImages(train_cache, train_pos, labels)
    best = fit(cfg, ctx, model, train_ds, train_array, val_pos, labels[val_pos], log)

    use_amp, amp_dtype = amp_settings(cfg, ctx.device)
    _, eval_tf = make_transforms(cfg, ctx.device)
    val_probs = predict(model, eval_tf, train_array, val_pos, cfg, ctx, use_amp, amp_dtype, tta=cfg.tta_hflip)
    val_labels = labels[val_pos]
    metrics = log_validation(f"val best epoch={best['epoch']}" + (" TTA hflip" if cfg.tta_hflip else ""),
                             val_labels, val_probs, log)
    val = {"best_epoch": best["epoch"], "f1": metrics["f1"], "acc": metrics["acc"], "loss": metrics["loss"],
           "per_class_f1": metrics["per_class_f1"], "f1_no_tta": best["f1"]}
    val_frame = pd.DataFrame(val_probs, columns=PROB_COLS)
    val_frame.insert(0, "group", groups[val_pos])
    val_frame.insert(0, "label", val_labels)
    val_frame.insert(0, "file_name", train_df.file_name.to_numpy()[val_pos])
    return model, val, val_frame


def _new_model(cfg: Config, ctx, log):
    if not ctx.is_main:
        ctx.barrier()
    model = build_model(cfg.arch, num_classes=NUM_CLASSES, pretrained=cfg.pretrained,
                        pretrained_dir=cfg.pretrained_dir, log=log)
    if ctx.is_main:
        ctx.barrier()
    model = model.to(ctx.device, memory_format=torch.channels_last if cfg.channels_last else torch.contiguous_format)
    torch.manual_seed(cfg.seed + ctx.rank)
    return model


if __name__ == "__main__":
    cli(Config, run, execute)
