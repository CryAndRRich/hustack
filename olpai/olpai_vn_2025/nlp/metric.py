from __future__ import annotations

import csv
from pathlib import Path

from nlp.data import Group


def _f1(tp: int, fp: int, fn: int) -> dict:
    return {"tp": tp, "fp": fp, "fn": fn, "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0, "f1": 2 * tp / (2 * tp + fp + fn) if tp else 0.0}


def edges(groups: list[Group], include_unaligned: bool = False) -> set[tuple[int, int]]:
    if not include_unaligned:
        return {(e, v) for en, vi in groups for e in en for v in vi}
    return {(e, v) for en, vi in groups for e in (en or (-1,)) for v in (vi or (-1,))}


def edge_counts(gold: list[Group], pred: list[Group], include_unaligned: bool = False) -> tuple[int, int, int]:
    g, p = edges(gold, include_unaligned), edges(pred, include_unaligned)
    return len(g & p), len(p - g), len(g - p)


def score(gold: list[Group], pred: list[Group]) -> dict:
    return {"edge": _f1(*edge_counts(gold, pred)), "edge_all": _f1(*edge_counts(gold, pred, include_unaligned=True))}


def total(counts: list[tuple[int, int, int]]) -> dict:
    return _f1(*(sum(c[i] for c in counts) for i in range(3)))


def cell(ids: tuple[int, ...]) -> str:
    return ";".join(str(i) for i in sorted(ids)) if ids else "-1"


def sort_groups(groups: list[Group]) -> list[Group]:
    return sorted(groups, key=lambda g: (0, min(g[0])) if g[0] else (1, min(g[1])))


def write_submission(path: Path, groups: list[Group]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["id_en", "id_vn"])
        for en, vi in sort_groups(groups):
            writer.writerow([cell(en), cell(vi)])


def validate(groups: list[Group], en_ids: list[int], vi_ids: list[int]) -> None:
    en = [e for g in groups for e in g[0]]
    vi = [v for g in groups for v in g[1]]
    if any(not g[0] and not g[1] for g in groups):
        raise ValueError("group empty on both sides")
    if sorted(en) != sorted(en_ids) or sorted(vi) != sorted(vi_ids):
        raise ValueError(f"bad coverage: EN sentences {len(en)}/{len(en_ids)} (dupes {len(en) - len(set(en))}), "
                         f"VI sentences {len(vi)}/{len(vi_ids)} (dupes {len(vi) - len(set(vi))})")
