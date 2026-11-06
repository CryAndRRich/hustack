from __future__ import annotations

import random
from collections import Counter, defaultdict
from pathlib import Path


def read_lines(path: Path) -> list[str]:
    lines = Path(path).read_text(encoding="utf-8").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    return [line.rstrip("\r") for line in lines]


def clean(text: str) -> str:
    return " ".join(text.split())


def load_pairs(data_root: Path) -> list[tuple[str, str]]:
    zh = read_lines(Path(data_root) / "train" / "train.zh")
    vi = read_lines(Path(data_root) / "train" / "train.vi")
    if len(zh) != len(vi):
        raise ValueError(f"train.zh ({len(zh)} lines) and train.vi ({len(vi)} lines) mismatch")
    return [(clean(a), clean(b)) for a, b in zip(zh, vi)]


def load_test(data_root: Path, split: str) -> list[str]:
    return read_lines(Path(data_root) / split / f"{split}.zh")


def split_pairs(pairs: list[tuple[str, str]], val_size: int, dedupe: bool, seed: int, log=print):
    counts = Counter(zh for zh, _ in pairs)
    candidates = [i for i, (zh, _) in enumerate(pairs) if counts[zh] == 1 and zh]
    rng = random.Random(seed)
    val_idx = set(rng.sample(candidates, min(val_size, len(candidates))))
    val = [pairs[i] for i in sorted(val_idx)]
    train, seen = [], set()
    for i, pair in enumerate(pairs):
        if i in val_idx or not pair[0] or not pair[1]:
            continue
        if dedupe and pair in seen:
            continue
        seen.add(pair)
        train.append(pair)
    log(f"[data] {len(pairs)} pairs -> train {len(train)} (dedupe={dedupe}) | val {len(val)} "
        f"(unique zh sentences only, {len(candidates)} candidates)")
    return train, val


def build_memory(pairs: list[tuple[str, str]]) -> dict[str, str]:
    options = defaultdict(Counter)
    first = {}
    for i, (zh, vi) in enumerate(pairs):
        options[zh][vi] += 1
        first.setdefault((zh, vi), i)
    return {zh: max(c.items(), key=lambda kv: (kv[1], -first[(zh, kv[0])]))[0] for zh, c in options.items()}
