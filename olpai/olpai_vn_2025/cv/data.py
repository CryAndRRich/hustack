from __future__ import annotations

import hashlib
import io
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image

CACHE_VERSION = 1
DECODE_THREADS = 24


def decode(path: Path, size: int) -> np.ndarray:
    data = Path(path).read_bytes()
    with Image.open(io.BytesIO(data)) as image:
        image.draft("RGB", (size, size))
        image = image.convert("RGB")
        if image.size != (size, size):
            image = image.resize((size, size), Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.uint8)


def build_cache(data_root: Path, files: list[str], cache_dir: Path, size: int, log=print) -> Path:
    key = hashlib.sha1(json.dumps([files, size, CACHE_VERSION]).encode()).hexdigest()[:12]
    path = Path(cache_dir) / f"positives_{size}_{key}.npy"
    if path.exists():
        log(f"[cache] reusing {path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    tmp = path.with_suffix(".tmp.npy")
    array = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(len(files), size, size, 3))

    step = max(1, len(files) // 5)
    log(f"[cache] decoding {len(files)} storm images ({DECODE_THREADS} threads, {os.cpu_count()} CPU)...")

    def work(i: int) -> None:
        array[i] = decode(Path(data_root) / files[i], size)
        if (i + 1) % step == 0:
            elapsed = time.time() - started
            log(f"[cache]   ~{i + 1}/{len(files)} images, {elapsed:.0f}s ({(i + 1) / max(elapsed, 1e-6):.0f} img/s)")

    with ThreadPoolExecutor(max_workers=DECODE_THREADS) as pool:
        list(pool.map(work, range(len(files))))
    array.flush()
    del array
    os.replace(tmp, path)
    log(f"[cache] {len(files)} storm images -> {path} ({path.stat().st_size / 2**30:.2f} GB, "
        f"{time.time() - started:.0f}s)")
    return path


class ImageStore:
    def __init__(self, array: np.ndarray, device: torch.device, on_gpu: bool):
        self.array = array
        self.device = device
        self.data = None
        if on_gpu and device.type == "cuda" and len(array):
            self.data = torch.empty(array.shape, dtype=torch.uint8, device=device)
            for start in range(0, len(array), 256):
                self.data[start:start + 256].copy_(torch.from_numpy(np.ascontiguousarray(array[start:start + 256])))

    @property
    def location(self) -> str:
        return "VRAM" if self.data is not None else "RAM"

    def get(self, rows: torch.Tensor) -> torch.Tensor:
        flat = rows.reshape(-1)
        if self.data is not None:
            batch = self.data.index_select(0, flat.to(self.device))
        else:
            batch = torch.from_numpy(np.ascontiguousarray(self.array[flat.cpu().numpy()]))
            if self.device.type == "cuda":
                batch = batch.pin_memory().to(self.device, non_blocking=True)
        return batch.view(*rows.shape, *batch.shape[1:])
