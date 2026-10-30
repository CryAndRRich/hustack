import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

SIZE = 320
SPLITS = {"train": ("train", "train_mask"), "val": ("val", "val_mask"), "test": ("test", None)}
CACHE = Path("/tmp/seg_cache")
CHUNK = 100


def get_data_dir():
    kaggle = Path("/kaggle/input")
    if kaggle.exists():
        for p in sorted([*kaggle.glob("*/test_index.csv"), *kaggle.glob("*/*/test_index.csv")]):
            if (p.parent / "train_mask").is_dir():
                return p.parent
        raise FileNotFoundError("no test_index.csv next to train_mask/ within two levels of /kaggle/input")
    try:
        return Path(__file__).resolve().parent.parent / "data"
    except NameError:
        return Path.cwd() / "data"


def file_lists(d):
    test = pd.read_csv(d / "test_index.csv")
    assert (test["index"].to_numpy() == np.arange(len(test))).all(), "test_index.csv is not 0..N-1 in order"
    names = {"train": sorted(os.listdir(d / "train")), "val": sorted(os.listdir(d / "val")),
             "test": test["filename"].tolist()}
    for split in ("train", "val"):
        assert names[split] == sorted(os.listdir(d / SPLITS[split][1])), f"{split} images and masks differ"
    return names


def _fill(job):
    path, start, files, is_mask = job
    arr = np.load(path, mmap_mode="r+")
    for i, f in enumerate(files):
        if is_mask:
            a = cv2.imread(f, cv2.IMREAD_UNCHANGED)
            assert a is not None and a.shape == (SIZE, SIZE) and a.dtype == np.uint8, f
        else:
            a = cv2.imread(f, cv2.IMREAD_COLOR)
            assert a is not None and a.shape == (SIZE, SIZE, 3), f
            a = a[..., ::-1]
        arr[start + i] = a
    arr.flush()


def build_cache(d, names, workers=32):
    t0 = time.time()
    cv2.setNumThreads(1)
    CACHE.mkdir(parents=True, exist_ok=True)
    jobs = []
    for split, (img_dir, mask_dir) in SPLITS.items():
        for sub, is_mask in ((img_dir, False), (mask_dir, True)):
            if sub is None:
                continue
            path = CACHE / f"{split}_{'y' if is_mask else 'x'}.npy"
            shape = (len(names[split]), SIZE, SIZE) + (() if is_mask else (3,))
            np.lib.format.open_memmap(path, "w+", np.uint8, shape).flush()
            files = [str(d / sub / n) for n in names[split]]
            jobs += [(str(path), s, files[s:s + CHUNK], is_mask) for s in range(0, len(files), CHUNK)]
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(_fill, jobs))
    sizes = {p.name: p.stat().st_size / 2 ** 30 for p in sorted(CACHE.glob("*.npy"))}
    print(f"cache built in {time.time() - t0:.0f}s: " + ", ".join(f"{k} {v:.2f} GiB" for k, v in sizes.items()))
