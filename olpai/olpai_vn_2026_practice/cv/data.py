from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def frame_indices(n_frames: int, num_frames: int) -> np.ndarray:
    return np.linspace(0, max(n_frames - 1, 0), num_frames).round().astype(np.int64)


def is_solid(frame: np.ndarray, solid_std: float) -> bool:
    return float(frame[::4, ::4].reshape(-1, frame.shape[-1]).std(0).max()) < solid_std


def decode_video(path: Path, num_frames: int, size: int, solid_std: float = 0.0) -> tuple[np.ndarray, int, int]:
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    out = np.zeros((num_frames, size, size, 3), dtype=np.uint8)
    if not frames:
        return out, 0, 0
    total, solid = len(frames), 0
    if solid_std > 0:
        kept = [f for f in frames if not is_solid(f, solid_std)]
        solid = total - len(kept)
        if kept:
            frames = kept
    for t, i in enumerate(frame_indices(len(frames), num_frames)):
        frame = frames[i]
        if frame.shape[:2] != (size, size):
            frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
        out[t] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return out, total, solid


def cache_path(cache_dir: Path, tag: str, files: list[str], num_frames: int, size: int,
               solid_std: float = 0.0) -> Path:
    key = hashlib.sha1(json.dumps([files, num_frames, size, solid_std], ensure_ascii=False).encode()).hexdigest()[:12]
    return Path(cache_dir) / f"{tag}_{num_frames}x{size}_{key}.npy"


def build_cache(root: Path, files: list[str], cache_dir: Path, tag: str, num_frames: int, size: int,
                solid_std: float = 0.0, log=print) -> Path:
    path = cache_path(cache_dir, tag, files, num_frames, size, solid_std)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npy")
    array = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8,
                                      shape=(len(files), num_frames, size, size, 3))
    counts = np.zeros(len(files), dtype=np.int64)
    solids = np.zeros(len(files), dtype=np.int64)

    def decode(i: int) -> None:
        array[i], counts[i], solids[i] = decode_video(Path(root) / files[i], num_frames, size, solid_std)

    cv2.setNumThreads(1)
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as pool:
        list(pool.map(decode, range(len(files))))
    array.flush()
    del array
    os.replace(tmp, path)
    failed = [files[i] for i in np.flatnonzero(counts == 0)]
    log(f"[cache] {tag}: {len(files)} video -> {path} ({path.stat().st_size / 2**30:.2f} GB) | source frame count "
        f"min/median/max = {counts.min()}/{int(np.median(counts))}/{counts.max()}"
        + (f" | dropped solid-colour frames in {int((solids > 0).sum())} videos ({(solids > 0).mean():.1%}), "
           f"{int(solids.sum())}/{int(counts.sum())} frames; {int((solids == counts).sum())} videos fully solid (kept as is)"
           if solid_std > 0 else "")
        + (f" | READ ERROR in {len(failed)} videos (black frames): {failed[:5]}" if failed else ""))
    return path


def open_cache(cache_dir: Path, tag: str, files: list[str], num_frames: int, size: int, solid_std: float = 0.0):
    path = cache_path(cache_dir, tag, files, num_frames, size, solid_std)
    if not path.exists():
        raise FileNotFoundError(f"No cache at {path}; call build_cache first")
    return path, np.load(path, mmap_mode="r")


def temporal_window(num_frames: int, scale: tuple[float, float], rng: np.random.Generator) -> np.ndarray:
    length = max(1, int(round(num_frames * rng.uniform(*scale))))
    start = int(rng.integers(0, num_frames - length + 1))
    return np.linspace(start, start + length - 1, num_frames).round().astype(np.int64)


class CachedClips(Dataset):
    def __init__(self, cache_file: Path, indices, labels=None, temporal_scale=(1.0, 1.0)):
        self.cache_file = str(cache_file)
        self.indices = np.asarray(indices)
        self.labels = None if labels is None else np.asarray(labels)
        self.temporal_scale = tuple(temporal_scale)
        self._array = None
        self._rng = None

    @property
    def array(self) -> np.ndarray:
        if self._array is None:
            self._array = np.load(self.cache_file, mmap_mode="r")
        return self._array

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_array"] = None
        state["_rng"] = None
        return state

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        if self._rng is None:
            self._rng = np.random.default_rng(np.random.randint(2**31))
        index = int(self.indices[i])
        clip = self.array[index]
        if self.temporal_scale[0] < 1:
            clip = clip[temporal_window(len(clip), self.temporal_scale, self._rng)]
        label = int(self.labels[index]) if self.labels is not None else -1
        return torch.from_numpy(np.array(clip)), label, index


def load_batch(array: np.ndarray, indices: np.ndarray, pin: bool) -> torch.Tensor:
    batch = torch.from_numpy(np.ascontiguousarray(array[indices]))
    return batch.pin_memory() if pin else batch
