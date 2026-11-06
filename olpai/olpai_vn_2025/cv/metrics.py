from __future__ import annotations

import csv
from pathlib import Path

from cv.config import NUM_CLASSES

SUBMISSION_COLUMNS = ["file_name", "is_negative", "category_id"]


def argmax(values) -> int:
    return max(range(len(values)), key=values.__getitem__)


def positive_summary(labels: list[int], probs: list[list[float]]) -> dict:
    pred = [argmax(p) for p in probs]
    n = len(labels)
    acc = sum(p == t for p, t in zip(pred, labels)) / n if n else 0.0
    confusion = [[0] * NUM_CLASSES for _ in range(NUM_CLASSES)]
    for t, p in zip(labels, pred):
        confusion[t][p] += 1
    per_class = [round(confusion[k][k] / sum(confusion[k]), 4) if sum(confusion[k]) else None for k in range(NUM_CLASSES)]
    return {"n": n, "acc": acc, "f1": 2 * acc / (1 + acc) if acc else 0.0, "acc_per_class": per_class,
            "confusion": confusion}


def format_summary(tag: str, s: dict) -> str:
    rows = " ".join(f"{k + 1}:{row}" for k, row in enumerate(s["confusion"]))
    return (f"[{tag}] n={s['n']} acc={s['acc']:.4f} F1(task)={s['f1']:.4f} | per-category acc {s['acc_per_class']} "
            f"| confusion (true: predicted category 1..4) {rows}")


def write_submission(path: Path, names: list[str], pred: list[int]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(SUBMISSION_COLUMNS)
        for name, p in zip(names, pred):
            writer.writerow([name, "TRUE", ""] if p < 0 else [name, "FALSE", p + 1])


def validate_submission(path: Path, expected_names: list[str]) -> dict:
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        rows = list(reader)
    if header != SUBMISSION_COLUMNS:
        raise ValueError(f"{path.name}: wrong columns {header}")
    names = [r[0] for r in rows]
    if len(names) != len(set(names)):
        raise ValueError(f"{path.name}: duplicate file_name")
    if set(names) != set(expected_names):
        raise ValueError(f"{path.name}: image list mismatch (missing {len(set(expected_names) - set(names))}, "
                         f"extra {len(set(names) - set(expected_names))})")
    counts: dict[str, int] = {}
    for r in rows:
        if len(r) != 3 or (r[1], r[2] == "") not in (("TRUE", True), ("FALSE", False)) \
                or (r[1] == "FALSE" and r[2] not in ("1", "2", "3", "4")):
            raise ValueError(f"{path.name}: malformed row {r}")
        key = "N" if r[1] == "TRUE" else r[2]
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))
