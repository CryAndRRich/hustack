from __future__ import annotations

import hashlib
import pickletools
import random
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
_STR_OPS = {"SHORT_BINUNICODE", "BINUNICODE", "BINUNICODE8", "UNICODE"}
_INT_OPS = {"BININT", "BININT1", "BININT2", "INT", "LONG1"}
_UNSAFE_OPS = {"GLOBAL", "STACK_GLOBAL", "REDUCE", "BUILD", "INST", "OBJ", "NEWOBJ", "NEWOBJ_EX"}


def nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def read_label_mapping(path: Path) -> dict[str, int]:
    keys, values = [], []
    for op, arg, _ in pickletools.genops(Path(path).read_bytes()):
        if op.name in _UNSAFE_OPS:
            raise ValueError(f"{path}: pickle contains {op.name}, not a dict str -> int")
        if op.name in _STR_OPS:
            keys.append(arg)
        elif op.name in _INT_OPS:
            values.append(int(arg))
    mapping = dict(zip(keys, values))
    if len(keys) != len(values) or len(mapping) != len(keys) or sorted(values) != list(range(len(values))):
        raise ValueError(f"{path}: could not read a valid label mapping ({len(keys)} keys, {len(values)} indices)")
    return mapping


def class_names(data_root: Path) -> list[str]:
    mapping = read_label_mapping(Path(data_root) / "label_mapping.pkl")
    return [name for name, _ in sorted(mapping.items(), key=lambda kv: kv[1])]


def is_video(path: Path) -> bool:
    return path.is_file() and not path.name.startswith(".") and path.suffix.lower() in VIDEO_EXTS


def list_videos(data_root: Path, split: str) -> list[str]:
    folder = Path(data_root) / split
    if not folder.is_dir():
        raise FileNotFoundError(f"{folder} not found")
    return sorted(p.name for p in folder.iterdir() if is_video(p))


def md5_file(path: Path) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def scan_train(data_root: Path, names: list[str]) -> list[dict]:
    data_root = Path(data_root)
    index = {nfc(n): i for i, n in enumerate(names)}
    rows = []
    for folder in sorted((data_root / "train").iterdir()):
        if not folder.is_dir():
            continue
        if nfc(folder.name) not in index:
            raise ValueError(f"Class folder {folder.name!r} not in label_mapping.pkl")
        label = index[nfc(folder.name)]
        rows += [{"path": f"train/{folder.name}/{p.name}", "label": label}
                 for p in sorted(folder.iterdir()) if is_video(p)]
    found = {r["label"] for r in rows}
    if len(found) != len(names):
        raise ValueError(f"Train has {len(found)} classes, label_mapping has {len(names)}")
    with ThreadPoolExecutor(max_workers=8) as pool:
        for row, digest in zip(rows, pool.map(md5_file, (data_root / r["path"] for r in rows))):
            row["md5"] = digest
    return rows


def dedupe(rows: list[dict], log=print) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[row["md5"]].append(row)
    unique, dropped, conflicts = [], 0, 0
    for digest, members in groups.items():
        votes = Counter(m["label"] for m in members).most_common()
        if len(votes) > 1:
            conflicts += 1
            if votes[0][1] == votes[1][1]:
                dropped += 1
                continue
        first = min(members, key=lambda m: m["path"])
        unique.append({"path": first["path"], "label": votes[0][0], "md5": digest, "copies": len(members)})
    unique.sort(key=lambda r: r["path"])
    log(f"[MD5 dedup] {len(rows)} files -> {len(unique)} unique videos | {conflicts} groups with conflicting labels "
        f"(dropped {dropped} tied groups, majority label for the rest)")
    return unique


def stratified_split(labels: list[int], val_size: float, seed: int) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    by_class = defaultdict(list)
    for pos, label in enumerate(labels):
        by_class[label].append(pos)
    train, val = [], []
    for label in sorted(by_class):
        members = by_class[label][:]
        rng.shuffle(members)
        n_val = max(1, round(len(members) * val_size)) if len(members) >= 2 else 0
        val += members[:n_val]
        train += members[n_val:]
    return sorted(train), sorted(val)


def flip_permutation(names: list[str], pairs) -> list[int]:
    index = {nfc(n): i for i, n in enumerate(names)}
    perm = list(range(len(names)))
    for a, b in pairs:
        missing = [x for x in (a, b) if nfc(x) not in index]
        if missing:
            raise ValueError(f"flip_pairs has unknown classes: {missing}")
        i, j = index[nfc(a)], index[nfc(b)]
        perm[i], perm[j] = j, i
    return perm
