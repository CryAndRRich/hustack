from __future__ import annotations

import hashlib
import io
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from cv.config import SOURCE_SIZE as CACHE_SIZE

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def load_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def load_train(data_root: Path) -> pd.DataFrame:
    df = pd.read_csv(data_root / "train" / "manifest.csv")
    df["file_name"] = df["path"].map(lambda p: Path(p).name)
    return df[["file_name", "label"]].astype({"label": int}).reset_index(drop=True)


def list_images(data_root: Path, split: str) -> list[str]:
    image_dir = data_root / split / "images"
    if not image_dir.is_dir():
        hint = " (private_test.zip not extracted?)" if (data_root / f"{split}.zip").exists() else ""
        raise FileNotFoundError(f"{image_dir} not found{hint}")
    return sorted(
        p.name for p in image_dir.iterdir()
        if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in IMAGE_EXTS
    )


def cache_path(cache_dir: Path, image_dir: Path, names: list[str]) -> Path:
    key = hashlib.sha1(json.dumps([str(image_dir), CACHE_SIZE, names]).encode()).hexdigest()[:12]
    return Path(cache_dir) / f"{image_dir.parent.name}_{CACHE_SIZE}_{key}.npy"


def build_cache(image_dir: Path, names: list[str], cache_dir: Path, log=print) -> Path:
    path = cache_path(cache_dir, image_dir, names)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npy")
    array = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(len(names), CACHE_SIZE, CACHE_SIZE, 3))

    def decode(i: int) -> None:
        image = load_image(image_dir / names[i])
        if image.size != (CACHE_SIZE, CACHE_SIZE):
            image = image.resize((CACHE_SIZE, CACHE_SIZE), Image.BICUBIC)
        array[i] = np.asarray(image)

    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as pool:
        list(pool.map(decode, range(len(names))))
    array.flush()
    del array
    os.replace(tmp, path)
    log(f"[cache] {len(names)} images {image_dir} -> {path}")
    return path


def open_cache(cache_dir: Path, image_dir: Path, names: list[str]):
    path = cache_path(cache_dir, image_dir, names)
    if not path.exists():
        raise FileNotFoundError(f"Cache {path} missing; call build_cache first")
    return path, np.load(path, mmap_mode="r")


class RandomJPEG:
    def __init__(self, p: float = 0.3):
        self.quality = (60, 100)
        self.p = p

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if random.random() >= self.p:
            return image
        buffer = io.BytesIO()
        Image.fromarray(image).save(buffer, format="JPEG", quality=random.randint(*self.quality))
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            return np.array(decoded.convert("RGB"))


class CachedImages(Dataset):
    def __init__(self, cache_file: Path, indices, labels=None, cpu_transform=None):
        self.cache_file = str(cache_file)
        self.indices = np.asarray(indices)
        self.labels = None if labels is None else np.asarray(labels)
        self.cpu_transform = cpu_transform
        self._array = None

    @property
    def array(self) -> np.ndarray:
        if self._array is None:
            self._array = np.load(self.cache_file, mmap_mode="r")
        return self._array

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_array"] = None
        return state

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        index = int(self.indices[i])
        image = np.array(self.array[index])
        if self.cpu_transform is not None:
            image = self.cpu_transform(image)
        label = int(self.labels[index]) if self.labels is not None else -1
        return torch.from_numpy(image), label, index


def load_batch(array: np.ndarray, indices: np.ndarray, pin: bool) -> torch.Tensor:
    batch = torch.from_numpy(np.ascontiguousarray(array[indices]))
    return batch.pin_memory() if pin else batch
