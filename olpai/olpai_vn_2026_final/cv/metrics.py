from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

from cv.config import CLASSES

NUM_CLASSES = len(CLASSES)


def macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, labels=range(NUM_CLASSES), average="macro", zero_division=0))


def per_class_f1(y_true, y_pred) -> dict[str, float]:
    scores = f1_score(y_true, y_pred, labels=range(NUM_CLASSES), average=None, zero_division=0)
    return {name: round(float(s), 4) for name, s in zip(CLASSES, scores)}


def log_loss(y_true, probs) -> float:
    p = np.clip(np.asarray(probs, dtype=np.float64)[np.arange(len(y_true)), np.asarray(y_true)], 1e-7, 1)
    return float(-np.log(p).mean())


def summarize(y_true, probs) -> dict:
    pred = np.asarray(probs).argmax(1)
    return {
        "f1": macro_f1(y_true, pred),
        "acc": float((pred == np.asarray(y_true)).mean()),
        "loss": log_loss(y_true, probs),
        "per_class_f1": per_class_f1(y_true, pred),
    }


def confusion_table(y_true, y_pred) -> pd.DataFrame:
    matrix = confusion_matrix(y_true, y_pred, labels=range(NUM_CLASSES))
    return pd.DataFrame(matrix, index=CLASSES, columns=CLASSES)


def validate_submission(submission: pd.DataFrame, expected_names: list[str]) -> None:
    if list(submission.columns) != ["file_name", "category_id"]:
        raise ValueError(f"Wrong column names/order: {list(submission.columns)}")
    if submission.file_name.duplicated().any():
        raise ValueError("Duplicate file_name")
    if set(submission.file_name) != set(expected_names):
        missing = set(expected_names) - set(submission.file_name)
        extra = set(submission.file_name) - set(expected_names)
        raise ValueError(f"Image list mismatch: missing {len(missing)}, extra {len(extra)}")
    if not submission.category_id.isin(range(1, NUM_CLASSES + 1)).all():
        raise ValueError(f"category_id has values outside 1..{NUM_CLASSES}")
