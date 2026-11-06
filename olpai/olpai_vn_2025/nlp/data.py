from __future__ import annotations

import csv
from pathlib import Path

from nlp.config import SPLITS

Group = tuple[tuple[int, ...], tuple[int, ...]]


def _read(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def read_sentences(data_root: Path, split: str) -> tuple[dict[int, str], dict[int, str]]:
    root = Path(data_root) / split
    en = {int(r["id_en"]): r["english_sentence"].strip() for r in _read(root / "en.csv")}
    vi = {int(r["id_vn"]): r["vietnamese_sentence"].strip() for r in _read(root / "vi.csv")}
    return en, vi


def parse_ids(cell: str) -> tuple[int, ...]:
    ids = tuple(int(x) for x in cell.strip().split(";"))
    return () if ids == (-1,) else ids


def read_alignment(data_root: Path) -> list[Group]:
    return [(parse_ids(r["id_en"]), parse_ids(r["id_vn"])) for r in _read(Path(data_root) / "train" / "correct_alignment.csv")]


def make_pools(data_root: Path, blocks: int) -> list[dict]:
    pools = []
    for split in SPLITS:
        en, vi = read_sentences(data_root, split)
        pools.append({"name": split, "split": split, "en": sorted(en), "vi": sorted(vi)})
    groups = read_alignment(data_root)
    aligned = sorted((g for g in groups if g[0] and g[1]), key=lambda g: min(g[0]))
    lonely_en = sorted(e for g in groups if not g[1] for e in g[0])
    lonely_vi = sorted(v for g in groups if not g[0] for v in g[1])
    total = sum(len(g[0]) for g in aligned)
    cuts, count, k = [], 0, 1
    for i, g in enumerate(aligned):
        count += len(g[0])
        if k < blocks and count >= total * k / blocks:
            cuts.append(i + 1)
            k += 1
    bounds = [0] + cuts + [len(aligned)]
    for b in range(blocks):
        part = aligned[bounds[b]:bounds[b + 1]]
        pools.append({"name": f"val{b}", "split": "train",
                      "en": sorted([e for g in part for e in g[0]] + lonely_en[b::blocks]),
                      "vi": sorted([v for g in part for v in g[1]] + lonely_vi[b::blocks])})
    return pools


def pool_gold(pool: dict, groups: list[Group]) -> list[Group]:
    en, vi = set(pool["en"]), set(pool["vi"])
    out = [g for g in groups if (g[0] and set(g[0]) <= en) or (not g[0] and set(g[1]) <= vi)]
    if {e for g in out for e in g[0]} != en or {v for g in out for v in g[1]} != vi:
        raise ValueError(f"gold does not cover pool {pool['name']} exactly")
    return out
