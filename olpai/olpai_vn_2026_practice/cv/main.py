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
from common.training import amp_settings
from common.utils import Logger, seed_everything
from cv.augment import ClipAugment, tta_views
from cv.catalog import class_names, dedupe, flip_permutation, list_videos, nfc, scan_train, stratified_split
from cv.config import NUM_CLASSES, Config, find_data_root, resolve_output_dir
from cv.data import CachedClips, build_cache, open_cache
from cv.engine import fit, predict
from cv.metrics import summarize, top_confusions, validate_submission, worst_classes
from cv.model import build_video_model

TRAIN_TABLE = "train_unique.csv"
SUBMISSION_CSV = "cv_submission.csv"
SUBMISSION_ZIP = "cv_submission.zip"
PROB_COLS = [f"p_{i:03d}" for i in range(NUM_CLASSES)]


def prepare(cfg: Config) -> Config:
    data_root = find_data_root(cfg.data_root)
    output_dir = resolve_output_dir(cfg.output_dir)
    cfg, output_dir = cfg.for_run(output_dir)
    cache_dir = cfg.cache_dir_or("olpai_vn_2026_practice_cv_cache")
    return replace(cfg, data_root=str(data_root), output_dir=str(output_dir), cache_dir=str(cache_dir))


def test_files(cfg: Config) -> list[str]:
    names = list_videos(Path(cfg.data_root), cfg.split)
    return names[:16] if cfg.debug else names


def prepare_train(cfg: Config, log) -> pd.DataFrame:
    data_root = Path(cfg.data_root)
    names = class_names(data_root)
    rows = dedupe(scan_train(data_root, names), log)
    df = pd.DataFrame(rows)
    if cfg.debug:
        df = df.groupby("label").head(3).reset_index(drop=True)
    _, val_pos = stratified_split(df.label.tolist(), cfg.val_size, cfg.seed)
    df["is_val"] = False
    df.loc[val_pos, "is_val"] = True
    build_cache(data_root, df.path.tolist(), Path(cfg.cache_dir), "train", cfg.num_frames, cfg.image_size,
                cfg.solid_std, log=log)
    df.to_csv(Path(cfg.output_dir) / TRAIN_TABLE, index=False)
    return df


def run(cfg: Config | None = None) -> dict:
    cfg = prepare(cfg or Config())
    output_dir = Path(cfg.output_dir)
    log = Logger(output_dir / "train.log")

    if not cfg.checkpoint:
        prepare_train(cfg, log)
    files = [f"{cfg.split}/{n}" for n in test_files(cfg)]
    build_cache(Path(cfg.data_root), files, Path(cfg.cache_dir), cfg.split, cfg.num_frames, cfg.image_size,
                cfg.solid_std, log=log)

    return run_or_launch(cfg, "cv.main", execute, log)


def execute(cfg: Config) -> dict | None:
    return run_ranks(_execute, cfg)


def _execute(cfg: Config, ctx) -> dict | None:
    output_dir = Path(cfg.output_dir)
    log = Logger(output_dir / "train.log", enabled=ctx.is_main)
    if cfg.num_workers is None:
        cfg = replace(cfg, num_workers=min(4, max(1, (os.cpu_count() or 2) // ctx.world_size)))
    seed_everything(cfg.seed)

    device = ctx.device
    gpu = f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""
    log(f"[config] {json.dumps(asdict(cfg), ensure_ascii=False)}")
    log(f"[config] device={device}{gpu} × {ctx.world_size} | torch {torch.__version__}")

    names = class_names(Path(cfg.data_root))
    swap = torch.tensor(flip_permutation(names, cfg.flip_pairs))
    test_names = test_files(cfg)
    _, test_array = open_cache(Path(cfg.cache_dir), cfg.split, [f"{cfg.split}/{n}" for n in test_names],
                               cfg.num_frames, cfg.image_size, cfg.solid_std)
    log(f"[test] {cfg.split}: {len(test_names)} video" + (" — DEBUG: RESULTS NOT FOR SUBMISSION" if cfg.debug else ""))

    if cfg.checkpoint:
        ckpt = torch.load(cfg.checkpoint, map_location="cpu", weights_only=False)
        keep = {k: ckpt["config"][k] for k in ("arch", "tsm", "tsm_fold_div", "temporal_dim", "temporal_layers",
                                                "temporal_heads", "dropout", "num_frames", "image_size")}
        if (keep["num_frames"], keep["image_size"]) != (cfg.num_frames, cfg.image_size):
            raise ValueError(f"checkpoint needs num_frames/image_size = {keep['num_frames']}/{keep['image_size']}")
        cfg = replace(cfg, **keep)
        model = build_video_model(cfg, NUM_CLASSES, pretrained=False, log=log)
        model.load_state_dict(ckpt["state_dict"])
        model = model.to(device, memory_format=torch.channels_last if cfg.channels_last else torch.contiguous_format)
        val, val_frames = ckpt.get("val", {}), None
        log(f"[checkpoint] {cfg.checkpoint} | {cfg.arch} (TSM={cfg.tsm})")
    else:
        model, val, val_frames = _train(cfg, ctx, names, swap, log)

    use_amp, amp_dtype = amp_settings(cfg, device)
    transform = ClipAugment(cfg, device, swap)
    test_probs, info = predict(model, transform, test_array, np.arange(len(test_names)), cfg, ctx, use_amp,
                               amp_dtype, views=tta_views(cfg), return_info=True)
    if not ctx.is_main:
        return None

    bad = ~np.isfinite(test_probs).all(1)
    log(f"[test] {cfg.split}: median filter (noise detected) {int(info['noisy'].sum())}/{len(test_names)} videos "
        f"| batches recomputed in fp32: {info['fp32_fallback']} | non-finite rows: {int(bad.sum())}")
    if bad.any():
        log(f"[test] WARNING: {int(bad.sum())} videos without finite probabilities -> not writing this run's submission "
            f"(ensemble skips this run on those videos): {[test_names[i] for i in np.flatnonzero(bad)[:5]]}")
        csv_path = zip_path = None
    else:
        csv_path, zip_path = write_submission(test_names, test_probs, names, output_dir, log)
    test_frame = pd.DataFrame(test_probs, columns=PROB_COLS)
    test_frame.insert(0, "denoised", info["noisy"])
    test_frame.insert(0, "video_name", test_names)
    test_frame.to_csv(output_dir / f"{cfg.split}_probs.csv", index=False)
    if not cfg.checkpoint:
        torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                    "config": asdict(cfg), "classes": names, "val": val}, output_dir / "best_model.pt")
        for name, frame in val_frames.items():
            frame.to_csv(output_dir / name, index=False)

    result = {"val": val, "submission_csv": csv_path and str(csv_path), "submission_zip": zip_path and str(zip_path),
              "test_denoised": int(info["noisy"].sum()), "test_nonfinite": int(bad.sum()),
              "output_dir": str(output_dir), "config": asdict(cfg)}
    (output_dir / RESULT_FILE).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def write_submission(videos: list[str], probs: np.ndarray, names: list[str], output_dir: Path, log=print):
    if not np.isfinite(probs).all():
        raise ValueError(f"{int((~np.isfinite(probs).all(1)).sum())} rows with non-finite probabilities — not writing submission")
    submission = pd.DataFrame({"video_name": videos, "label": [names[i] for i in probs.argmax(1)]})
    validate_submission(submission, videos, names)
    csv_path, zip_path = output_dir / SUBMISSION_CSV, output_dir / SUBMISSION_ZIP
    submission.to_csv(csv_path, index=False, encoding="utf-8")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(csv_path, arcname=SUBMISSION_CSV)
    counts = np.bincount(probs.argmax(1), minlength=NUM_CLASSES)
    log(f"[submit] {zip_path} | {len(submission)} videos | classes predicted "
        f"{int((counts > 0).sum())}/{NUM_CLASSES} | max {int(counts.max())}, min {int(counts.min())} videos/class"
        f" | {confidence_summary(probs)}")
    return csv_path, zip_path


def confidence_summary(probs: np.ndarray) -> str:
    top = probs.max(1)
    return f"median confidence {np.median(top):.3f}, share < 0.5: {(top < 0.5).mean():.1%}"


def log_validation(tag: str, labels: np.ndarray, probs: np.ndarray, names: list[str], log) -> dict:
    metrics = summarize(labels, probs)
    pred = probs.argmax(1)
    shown = [nfc(n) for n in names]
    log(f"[{tag}] Macro F1={metrics['f1']:.4f} | acc={metrics['acc']:.4f} | loss={metrics['loss']:.4f}"
        f"\n  10 lowest-F1 classes: {worst_classes(labels, pred, shown)}"
        f"\n  10 most confused pairs (true -> pred, videos): {top_confusions(labels, pred, shown)}")
    return metrics


def _train(cfg: Config, ctx, names: list[str], swap, log):
    data_root, cache_dir = Path(cfg.data_root), Path(cfg.cache_dir)
    df = pd.read_csv(Path(cfg.output_dir) / TRAIN_TABLE)
    cache_file, array = open_cache(cache_dir, "train", df.path.tolist(), cfg.num_frames, cfg.image_size, cfg.solid_std)
    labels = df.label.to_numpy()
    train_pos = np.flatnonzero(~df.is_val.to_numpy())
    val_pos = np.flatnonzero(df.is_val.to_numpy())
    per_class = np.bincount(labels[train_pos], minlength=NUM_CLASSES)
    log(f"[data] {data_root} | train={len(train_pos)} videos (per class {per_class.min()}–{per_class.max()}) "
        f"| val={len(val_pos)} videos ({len(np.unique(labels[val_pos]))} classes) | split after MD5 dedup")

    model = _new_model(cfg, ctx, log)
    train_ds = CachedClips(cache_file, train_pos, labels, temporal_scale=cfg.temporal_scale)
    best = fit(cfg, ctx, model, train_ds, array, val_pos, labels[val_pos], swap, log)

    use_amp, amp_dtype = amp_settings(cfg, ctx.device)
    transform = ClipAugment(cfg, ctx.device, swap)
    views = tta_views(cfg)
    val_probs, info = predict(model, transform, array, val_pos, cfg, ctx, use_amp, amp_dtype, views=views,
                              return_info=True)
    val_labels = labels[val_pos]
    metrics = log_validation(f"val best epoch={best['epoch']} TTA {views}", val_labels, val_probs, names, log)
    log(f"[val] no-TTA F1={best['f1']:.4f} -> TTA F1={metrics['f1']:.4f} | {confidence_summary(val_probs)} "
        f"| median filter false positives {int(info['noisy'].sum())}/{len(val_pos)} clean clips")
    val = {"best_epoch": best["epoch"], "f1": metrics["f1"], "acc": metrics["acc"], "loss": metrics["loss"],
           "f1_no_tta": best["f1"], "clean_denoised": int(info["noisy"].sum())}

    def frame(probs: np.ndarray) -> pd.DataFrame:
        out = pd.DataFrame(probs, columns=PROB_COLS)
        out.insert(0, "label", val_labels)
        out.insert(0, "path", df.path.to_numpy()[val_pos])
        return out

    frames = {"val_preds.csv": frame(val_probs)}
    if cfg.noisy_val:
        noisy_probs, noisy_info = predict(model, transform, array, val_pos, cfg, ctx, use_amp, amp_dtype,
                                          views=views, noise_seed=cfg.seed, return_info=True)
        noisy = log_validation(f"val WITH SYNTHETIC NOISE density {cfg.noise_density}", val_labels, noisy_probs, names, log)
        log(f"[val noisy] Macro F1 {noisy['f1']:.4f} (clean {metrics['f1']:.4f}) | noise detected "
            f"{int(noisy_info['noisy'].sum())}/{len(val_pos)} clip")
        val.update(f1_noisy=noisy["f1"], noisy_detected=int(noisy_info["noisy"].sum()))
        frames["val_noisy_preds.csv"] = frame(noisy_probs)
    return model, val, frames


def _new_model(cfg: Config, ctx, log):
    if not ctx.is_main:
        ctx.barrier()
    model = build_video_model(cfg, NUM_CLASSES, log=log)
    if ctx.is_main:
        ctx.barrier()
    model = model.to(ctx.device, memory_format=torch.channels_last if cfg.channels_last else torch.contiguous_format)
    torch.manual_seed(cfg.seed + ctx.rank)
    return model


if __name__ == "__main__":
    cli(Config, run, execute)
