from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

CROP_INSET = (4, 3)
GRAY_MEAN, GRAY_STD = 0.449, 0.226


def load_gray(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        if image.format == "JPEG":
            image.draft("L", image.size)
        return np.asarray(ImageOps.exif_transpose(image).convert("L"))


def cell_region(gray: np.ndarray, bbox) -> np.ndarray:
    h, w = gray.shape
    ix, iy = CROP_INSET
    x1, y1 = int(bbox[0] * w) + ix, int(bbox[1] * h) + iy
    x2, y2 = max(x1 + 1, int(bbox[2] * w) - ix), max(y1 + 1, int(bbox[3] * h) - iy)
    return gray[y1:y2, x1:x2]


def crop_cell(gray: np.ndarray, bbox, height: int, width: int) -> np.ndarray:
    region = cell_region(gray, bbox)
    new_w = max(1, round(region.shape[1] * height / max(1, region.shape[0])))
    resized = np.asarray(Image.fromarray(region).resize((new_w, height), Image.BILINEAR))
    out = np.full((height, width), 255, dtype=np.uint8)
    out[:, :min(new_w, width)] = resized[:, :width]
    return out


def build_crop_cache(pages: list[tuple[Path, list[tuple[str, tuple]]]], cache_dir: Path, name: str,
                     height: int, width: int, workers: int | None, log=print) -> tuple[Path, list[str]]:
    keys = [key for _, cells in pages for key, _ in cells]
    digest = hashlib.sha1(json.dumps([name, height, width, keys]).encode()).hexdigest()[:12]
    path = Path(cache_dir) / f"{name}_{height}x{width}_{digest}.npy"
    if path.exists():
        return path, keys
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npy")
    array = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(len(keys), height, width))
    offsets = np.cumsum([0] + [len(cells) for _, cells in pages])
    started = time.time()

    def work(i: int) -> None:
        image, cells = pages[i]
        gray = load_gray(image)
        for j, (_, bbox) in enumerate(cells):
            array[offsets[i] + j] = crop_cell(gray, bbox, height, width)

    with ThreadPoolExecutor(max_workers=workers or os.cpu_count() or 4) as pool:
        list(pool.map(work, range(len(pages))))
    array.flush()
    del array
    os.replace(tmp, path)
    log(f"[crop] {name}: {len(keys)} cells from {len(pages)} pages -> {path.name} ({time.time() - started:.0f}s)")
    return path, keys


class StrokeScorer:
    def __init__(self):
        self._pages: dict[tuple[str, int], np.ndarray] = {}

    def _gray(self, doc, page: int) -> np.ndarray:
        key = (doc.doc_id, page)
        if key not in self._pages:
            self._pages[key] = load_gray(doc.page(page).image_path)
        return self._pages[key]

    def __call__(self, doc, table, row: int) -> float:
        gray = self._gray(doc, table.page)
        runs = []
        for block in table.blocks:
            if block.row0 != row:
                continue
            dark = cell_region(gray, block.bbox) < 128
            if dark.size == 0:
                continue
            edges = np.diff(np.pad(dark.astype(np.int8), ((0, 0), (1, 1))), axis=1)
            lengths = np.where(edges == -1)[1] - np.where(edges == 1)[1]
            if len(lengths):
                runs.append(lengths.mean())
        return float(np.mean(runs)) if runs else 0.0


class CellScoreScorer:
    def __init__(self, scores: dict[str, float]):
        self.scores = scores

    def __call__(self, doc, table, row: int) -> float:
        values = [self.scores[f"{doc.doc_id}|{b.block_id}"] for b in table.blocks
                  if b.row0 == row and f"{doc.doc_id}|{b.block_id}" in self.scores]
        return float(np.mean(values)) if values else 0.0


def _torch():
    import torch
    return torch


class CropAugment:
    def __init__(self, device, train: bool, channels_last: bool = True):
        self.device, self.train, self.channels_last = device, train, channels_last
        self.shift, self.contrast, self.brightness, self.noise = 16, 0.2, 12.0, 4.0

    def __call__(self, images):
        torch = _torch()
        x = images.to(self.device, non_blocking=True).float()
        if self.train:
            n, h, w = x.shape
            k = torch.randint(0, self.shift + 1, (n, 1), device=x.device)
            cols = torch.arange(w, device=x.device).unsqueeze(0) - k
            x = torch.where((cols >= 0).unsqueeze(1), x.gather(2, cols.clamp(min=0).unsqueeze(1).expand(n, h, w)),
                            torch.full_like(x, 255.0))
            c = 1 + (torch.rand(n, 1, 1, device=x.device) * 2 - 1) * self.contrast
            b = (torch.rand(n, 1, 1, device=x.device) * 2 - 1) * self.brightness
            x = 255 - (255 - x) * c + b + torch.randn_like(x) * self.noise
            x = x.clamp(0, 255)
        x = ((x / 255 - GRAY_MEAN) / GRAY_STD).unsqueeze(1).expand(-1, 3, -1, -1)
        torch_format = torch.channels_last if self.channels_last else torch.contiguous_format
        return x.contiguous(memory_format=torch_format)


def _predict_logits(model, transform, array, cfg, ctx, use_amp, amp_dtype):
    torch = _torch()
    from common.training import autocast

    positions = np.arange(array.shape[0])[ctx.rank::ctx.world_size]
    batch = cfg.batch_size * 2
    with torch.inference_mode():
        model.eval()
        local = []
        for start in range(0, len(positions), batch):
            index = torch.as_tensor(positions[start:start + batch], device=array.device)
            with autocast(ctx.device, use_amp, amp_dtype):
                local.append(model(transform(array[index])).float().squeeze(1))
        local = torch.cat(local).cpu().numpy() if local else np.empty(0, dtype=np.float32)
    logits = np.empty(array.shape[0], dtype=np.float32)
    for pos, part in ctx.gather((positions, local)):
        logits[pos] = part
    return logits


def _metrics(labels: np.ndarray, logits: np.ndarray) -> dict:
    from sklearn.metrics import roc_auc_score
    p = 1 / (1 + np.exp(-logits))
    eps = 1e-7
    loss = float(-np.mean(labels * np.log(p + eps) + (1 - labels) * np.log(1 - p + eps)))
    acc = float(((p >= 0.5) == (labels >= 0.5)).mean())
    try:
        auc = float(roc_auc_score(labels, logits))
    except ValueError:
        auc = float("nan")
    return {"loss": loss, "acc": acc, "auc": auc}


def train_classifier(cfg, ctx, model, train_x, train_y, val_x, val_y, log) -> dict:
    torch = _torch()
    from torch import nn
    from torch.nn.parallel import DistributedDataParallel
    from common.training import (amp_settings, autocast, cosine_with_warmup, make_grad_scaler,
                                     make_optimizer, maybe_compile)

    device = ctx.device
    use_amp, amp_dtype = amp_settings(cfg, device)
    train_tf, eval_tf = CropAugment(device, True, cfg.channels_last), CropAugment(device, False, cfg.channels_last)
    wrapped = (DistributedDataParallel(model, device_ids=[ctx.local_rank] if device.type == "cuda" else None)
               if ctx.enabled else model)
    train_model = maybe_compile(wrapped, cfg, ctx, log)
    optimizer = make_optimizer(model, cfg, device)
    per_rank = train_x.shape[0] // ctx.world_size
    steps_per_epoch = max(1, per_rank // cfg.batch_size)
    scheduler = cosine_with_warmup(optimizer, steps_per_epoch * cfg.warmup_epochs, steps_per_epoch * cfg.epochs)
    scaler = make_grad_scaler(use_amp and amp_dtype == torch.float16)
    loss_fn = nn.BCEWithLogitsLoss()
    log(f"[bold] train {train_x.shape[0]} crop ({int(train_y.sum().item())} bold) | val {val_x.shape[0]} | "
        f"{ctx.world_size} procs × batch {cfg.batch_size} | {steps_per_epoch} steps/epoch | "
        f"AMP={'off' if not use_amp else str(amp_dtype).split('.')[-1]}")

    best = {"state": None, "epoch": 0, "loss": float("inf")}
    for epoch in range(1, cfg.epochs + 1):
        started = time.time()
        train_model.train()
        generator = torch.Generator(device="cpu").manual_seed(cfg.seed * 1000 + epoch)
        order = torch.randperm(train_x.shape[0], generator=generator)[ctx.rank::ctx.world_size].to(device)
        loss_sum = torch.zeros((), device=device)
        for step in range(steps_per_epoch):
            index = order[step * cfg.batch_size:(step + 1) * cfg.batch_size]
            x, y = train_tf(train_x[index]), train_y[index]
            with autocast(device, use_amp, amp_dtype):
                loss = loss_fn(train_model(x).float().squeeze(1), y)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            loss_sum += loss.detach()
        train_loss = (ctx.sum(loss_sum) / (steps_per_epoch * ctx.world_size)).item()
        train_time = time.time() - started

        ctx.broadcast_buffers(model)
        metrics = _metrics(val_y, _predict_logits(model, eval_tf, val_x, cfg, ctx, use_amp, amp_dtype))
        improved = metrics["loss"] < best["loss"]
        if improved:
            best.update(metrics, epoch=epoch, state={k: v.detach().clone() for k, v in model.state_dict().items()})
        log(f"[bold epoch {epoch}/{cfg.epochs}] train_loss={train_loss:.4f} | val_loss={metrics['loss']:.4f} "
            f"| val_acc={metrics['acc']:.4f} | val_auc={metrics['auc']:.4f} | {train_time:.1f}s "
            f"({steps_per_epoch * cfg.batch_size * ctx.world_size / train_time:.0f} crop/s)" + (" *" if improved else ""))
    model.load_state_dict(best.pop("state"))
    return best


def execute(job: dict) -> None:
    torch = _torch()
    from common.distributed import init_distributed
    from common.training import amp_settings
    from common.utils import Logger, seed_everything
    from common.vision import build_model
    from nlp.config import Config

    cfg = Config(**job["config"])
    ctx = init_distributed()
    try:
        out = Path(cfg.output_dir)
        log = Logger(out / "train.log", enabled=ctx.is_main)
        seed_everything(cfg.seed, cfg.deterministic)
        device = ctx.device
        gpu = f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""
        log(f"[bold] device={device}{gpu} × {ctx.world_size} | arch={cfg.arch}")

        def to_device(path):
            return torch.from_numpy(np.load(path)).to(device)

        eval_x = to_device(job["eval_cache"])
        if cfg.checkpoint:
            ckpt = torch.load(cfg.checkpoint, map_location="cpu", weights_only=False)
            cfg = replace(cfg, arch=ckpt["arch"], crop_height=ckpt["crop_height"], crop_width=ckpt["crop_width"])
            model = build_model(cfg.arch, num_classes=1, pretrained=False, log=log)
            model.load_state_dict(ckpt["state_dict"])
            model = model.to(device)
            best = ckpt.get("val", {})
            log(f"[bold] loaded checkpoint {cfg.checkpoint}")
        else:
            if not ctx.is_main:
                ctx.barrier()
            model = build_model(cfg.arch, num_classes=1, pretrained=cfg.pretrained,
                                pretrained_dir=cfg.pretrained_dir, log=log)
            if ctx.is_main:
                ctx.barrier()
            memory_format = torch.channels_last if cfg.channels_last else torch.contiguous_format
            model = model.to(device, memory_format=memory_format)
            torch.manual_seed(cfg.seed + ctx.rank)
            train_x, val_x = to_device(job["train_cache"]), to_device(job["val_cache"])
            train_y = torch.from_numpy(np.load(job["train_labels"]).astype(np.float32)).to(device)
            val_y = np.load(job["val_labels"]).astype(np.float32)
            best = train_classifier(cfg, ctx, model, train_x, train_y, val_x, val_y, log)
            del train_x, train_y, val_x

        use_amp, amp_dtype = amp_settings(cfg, device)
        logits = _predict_logits(model, CropAugment(device, False, cfg.channels_last), eval_x, cfg, ctx,
                                 use_amp, amp_dtype)
        if ctx.is_main:
            np.save(job["scores_out"], logits)
            if not cfg.checkpoint:
                torch.save({"arch": cfg.arch, "crop_height": cfg.crop_height, "crop_width": cfg.crop_width,
                            "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                            "val": best, "config": job["config"]}, out / "bold_model.pt")
            log(f"[bold] scored {len(logits)} crops | best val epoch: {best}")
    finally:
        ctx.close()


if __name__ == "__main__":
    if sys.argv[1:2] != ["--config"]:
        raise SystemExit("Usage: python -m nlp.bold --config <bold_job.json>")
    execute(json.loads(Path(sys.argv[2]).read_text(encoding="utf-8")))
