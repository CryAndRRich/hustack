from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from cv.config import CLASSES, SOURCE_HW

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def load_image(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def load_train(data_root: Path) -> pd.DataFrame:
    df = pd.read_csv(data_root / "train" / "train.csv")
    df["label"] = df["category_id"].astype(int) - 1
    if not df.label.between(0, len(CLASSES) - 1).all():
        raise ValueError("train.csv has category_id outside 1..6")
    return df[["file_name", "label"]].reset_index(drop=True)


def clean_train(data_root: Path, min_copies: int, log=print) -> pd.DataFrame:
    df = load_train(data_root)
    image_dir = data_root / "train" / "images"

    def md5(name: str) -> str:
        return hashlib.md5((image_dir / name).read_bytes()).hexdigest()

    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as pool:
        df["md5"] = list(pool.map(md5, df.file_name))
    copies = df.md5.map(df.md5.value_counts())
    junk = copies >= min_copies
    removed = df[junk].groupby("label").size().reindex(range(len(CLASSES)), fill_value=0)
    kept = df[~junk].reset_index(drop=True)
    log(f"[clean] dropped {int(junk.sum())}/{len(df)} images with MD5 duplicated >= {min_copies} times "
        f"({df[junk].md5.nunique()} distinct files) | per class {dict(zip(CLASSES, removed.tolist()))}")
    log(f"[clean] kept {len(kept)} images | per class "
        f"{dict(zip(CLASSES, np.bincount(kept.label, minlength=len(CLASSES)).tolist()))}")
    return kept


def list_images(data_root: Path, split: str) -> list[str]:
    image_dir = data_root / split / "images"
    if not image_dir.is_dir():
        hint = " (private_test.zip not unzipped?)" if (data_root / f"{split}.zip").exists() else ""
        raise FileNotFoundError(f"{image_dir} not found{hint}")
    return sorted(
        p.name for p in image_dir.iterdir()
        if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in IMAGE_EXTS
    )


def cache_path(cache_dir: Path, image_dir: Path, names: list[str]) -> Path:
    key = hashlib.sha1(json.dumps([str(image_dir), list(SOURCE_HW), names]).encode()).hexdigest()[:12]
    return Path(cache_dir) / f"{image_dir.parent.name}_{SOURCE_HW[0]}x{SOURCE_HW[1]}_{key}.npy"


def build_cache(image_dir: Path, names: list[str], cache_dir: Path, log=print) -> Path:
    path = cache_path(cache_dir, image_dir, names)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npy")
    height, width = SOURCE_HW
    array = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(len(names), height, width, 3))

    def decode(i: int) -> None:
        image = load_image(image_dir / names[i])
        if image.size != (width, height):
            image = image.resize((width, height), Image.BICUBIC)
        array[i] = np.asarray(image)

    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as pool:
        list(pool.map(decode, range(len(names))))
    array.flush()
    del array
    os.replace(tmp, path)
    log(f"[cache] {len(names)} images {image_dir} -> {path} ({path.stat().st_size / 2**30:.2f} GB)")
    return path


def open_cache(cache_dir: Path, image_dir: Path, names: list[str]):
    path = cache_path(cache_dir, image_dir, names)
    if not path.exists():
        raise FileNotFoundError(f"Cache {path} missing; call build_cache first")
    return path, np.load(path, mmap_mode="r")


class CachedImages(Dataset):
    def __init__(self, cache_file: Path, indices, labels=None):
        self.cache_file = str(cache_file)
        self.indices = np.asarray(indices)
        self.labels = None if labels is None else np.asarray(labels)
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
        image = torch.from_numpy(np.array(self.array[index]))
        label = int(self.labels[index]) if self.labels is not None else -1
        return image, label, index


def load_batch(array: np.ndarray, indices: np.ndarray, pin: bool) -> torch.Tensor:
    batch = torch.from_numpy(np.ascontiguousarray(array[indices]))
    return batch.pin_memory() if pin else batch
