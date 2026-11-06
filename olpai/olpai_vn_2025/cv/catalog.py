from __future__ import annotations

import csv
import random
import re
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

from cv.config import NUM_CLASSES, SPLITS

NAME_RE = re.compile(r"^(\d{12})_(\d{4})_[0-9A-Za-z]+\.jpg$")
TIME_FORMAT = "%Y%m%d%H%M"
STEP_HOURS = 3
FIELDS = ["file_name", "split", "storm", "time", "width", "height", "negative", "label", "fold", "row"]
ALL_SPLITS = ("train",) + SPLITS
HEADER_BYTES = 65536
IO_THREADS = 32


def parse_name(name: str) -> tuple[str, str]:
    match = NAME_RE.match(name)
    if not match:
        raise ValueError(f"image name does not match time_stormid_hash.jpg: {name!r}")
    return match.group(2), match.group(1)


def parse_time(stamp: str) -> datetime:
    return datetime.strptime(stamp, TIME_FORMAT)


def jpeg_size(path: Path) -> tuple[int, int]:
    with open(path, "rb") as fh:
        data = fh.read(HEADER_BYTES)
        size = _sof_size(data)
        if size is None:
            data += fh.read()
            size = _sof_size(data)
    if size is None:
        raise ValueError(f"SOF marker not found in {path}")
    return size


def _sof_size(data: bytes) -> tuple[int, int] | None:
    if data[:2] != b"\xff\xd8":
        raise ValueError("not a JPEG")
    i = 2
    while i + 4 <= len(data):
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker == 0xFF:
            i += 1
            continue
        if marker in (0xD8, 0x01, 0x00) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        length = struct.unpack_from(">H", data, i + 2)[0]
        if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            if i + 9 > len(data):
                return None
            height, width = struct.unpack_from(">HH", data, i + 5)
            return width, height
        i += 2 + length
    return None


def read_annotations(data_root: Path) -> dict[str, int]:
    labels = {}
    with open(Path(data_root) / "train" / "annotations.csv", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            negative = row["is_negative"].strip().upper() == "TRUE"
            labels[row["file_name"].strip()] = -1 if negative else int(row["category_id"]) - 1
    return labels


def list_images(data_root: Path, split: str) -> list[str]:
    return sorted(p.name for p in (Path(data_root) / split / "images").iterdir() if p.suffix.lower() == ".jpg")


def is_negative_shape(width: int, height: int, aspect: float) -> bool:
    return width / height >= aspect


def scan(data_root: Path, negative_aspect: float, log=print) -> list[dict]:
    labels = read_annotations(data_root)
    files = []
    for split in ALL_SPLITS:
        names = list_images(data_root, split)
        if split == "train" and set(names) != set(labels):
            raise ValueError(f"annotations.csv mismatches train image folder: {len(set(names) ^ set(labels))} files")
        files += [(split, name) for name in names]
    log(f"[catalog] reading headers of {len(files)} images ({IO_THREADS} threads)...")
    started = time.time()
    step = max(1, len(files) // 5)

    def size_of(i: int) -> tuple[int, int]:
        split, name = files[i]
        out = jpeg_size(Path(data_root) / split / "images" / name)
        if (i + 1) % step == 0:
            elapsed = time.time() - started
            log(f"[catalog]   ~{i + 1}/{len(files)} headers, {elapsed:.0f}s ({(i + 1) / max(elapsed, 1e-6):.0f} img/s)")
        return out

    with ThreadPoolExecutor(max_workers=IO_THREADS) as pool:
        sizes = list(pool.map(size_of, range(len(files))))
    rows = []
    for (split, name), (width, height) in zip(files, sizes):
        storm, stamp = parse_name(name)
        rows.append({"file_name": name, "split": split, "storm": storm, "time": stamp, "width": width,
                     "height": height, "negative": is_negative_shape(width, height, negative_aspect),
                     "label": labels.get(name, -1), "fold": -1, "row": -1})
    train = [r for r in rows if r["split"] == "train"]
    wrong = [r["file_name"] for r in train if r["negative"] != (r["label"] < 0)]
    if wrong:
        raise ValueError(f"size rule (width/height >= {negative_aspect}) wrong on {len(wrong)} train images: {wrong[:5]}")
    summary = []
    for split in ALL_SPLITS:
        part = [r for r in rows if r["split"] == split]
        summary.append(f"{split} {len(part)} images ({sum(r['negative'] for r in part)} no-storm)")
    log(f"[catalog] {' | '.join(summary)} | rule width/height >= {negative_aspect} matches 100% of train labels "
        f"| {time.time() - started:.0f}s")
    return rows


def assign_folds(rows: list[dict], folds: int, seed: int) -> None:
    positives = [r for r in rows if r["split"] == "train" and not r["negative"]]
    counts: dict[str, list[int]] = {}
    for r in positives:
        counts.setdefault(r["storm"], [0] * NUM_CLASSES)[r["label"]] += 1
    storms = sorted(counts)
    random.Random(seed).shuffle(storms)
    storms.sort(key=lambda s: -sum(counts[s]))
    target = [sum(c[k] for c in counts.values()) / folds for k in range(NUM_CLASSES)]
    totals = [[0] * NUM_CLASSES for _ in range(folds)]
    fold_of = {}
    for storm in storms:
        def cost(f: int) -> float:
            return sum((totals[f][k] + counts[storm][k] - target[k]) ** 2 - (totals[f][k] - target[k]) ** 2
                       for k in range(NUM_CLASSES))
        best = min(range(folds), key=cost)
        fold_of[storm] = best
        totals[best] = [a + b for a, b in zip(totals[best], counts[storm])]
    for r in positives:
        r["fold"] = fold_of[r["storm"]]


def assign_rows(rows: list[dict]) -> list[dict]:
    positives = [r for split in ALL_SPLITS for r in rows if r["split"] == split and not r["negative"]]
    for i, r in enumerate(positives):
        r["row"] = i
    return positives


def neighbor_rows(positives: list[dict], offsets: tuple[int, ...]) -> list[list[int]]:
    index = {}
    for r in positives:
        index.setdefault(("train" if r["split"] == "train" else "test", r["storm"], r["time"]), r["row"])
    out = [[0] * len(offsets) for _ in positives]
    for r in positives:
        group, t = ("train" if r["split"] == "train" else "test"), parse_time(r["time"])
        for j, k in enumerate(offsets):
            stamp = (t + timedelta(hours=k * STEP_HOURS)).strftime(TIME_FORMAT)
            out[r["row"]][j] = index.get((group, r["storm"], stamp), r["row"])
    return out


def write_catalog(rows: list[dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow({**r, "negative": int(r["negative"])})


def read_catalog(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for key in ("width", "height", "label", "fold", "row"):
            r[key] = int(r[key])
        r["negative"] = r["negative"] == "1"
    return rows
