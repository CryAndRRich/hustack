from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

from cv.config import NUM_CLASSES

SUBMISSION_COLUMNS = ["video_name", "label"]


def macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, labels=range(NUM_CLASSES), average="macro", zero_division=0))


def per_class_f1(y_true, y_pred) -> np.ndarray:
    return f1_score(y_true, y_pred, labels=range(NUM_CLASSES), average=None, zero_division=0)


def log_loss(y_true, probs) -> float:
    eps = 1e-7
    p = np.clip(np.asarray(probs, dtype=np.float64)[np.arange(len(y_true)), np.asarray(y_true)], eps, 1)
    return float(-np.log(p).mean())


def summarize(y_true, probs) -> dict:
    pred = np.asarray(probs).argmax(1)
    return {
        "f1": macro_f1(y_true, pred),
        "acc": float((pred == np.asarray(y_true)).mean()),
        "loss": log_loss(y_true, probs),
    }


def worst_classes(y_true, y_pred, names: list[str]) -> list[tuple[str, float]]:
    k = 10
    scores = per_class_f1(y_true, y_pred)
    present = np.unique(y_true)
    order = present[np.argsort(scores[present])][:k]
    return [(names[i], round(float(scores[i]), 3)) for i in order]


def top_confusions(y_true, y_pred, names: list[str]) -> list[tuple[str, str, int]]:
    k = 10
    matrix = confusion_matrix(y_true, y_pred, labels=range(NUM_CLASSES))
    np.fill_diagonal(matrix, 0)
    flat = np.argsort(matrix, axis=None)[::-1][:k]
    pairs = [np.unravel_index(i, matrix.shape) for i in flat]
    return [(names[t], names[p], int(matrix[t, p])) for t, p in pairs if matrix[t, p] > 0]


def validate_submission(submission: pd.DataFrame, expected_names: list[str], labels: list[str]) -> None:
    if list(submission.columns) != SUBMISSION_COLUMNS:
        raise ValueError(f"Wrong column names/order: {list(submission.columns)}")
    if submission.video_name.duplicated().any():
        raise ValueError("Duplicate video_name")
    if set(submission.video_name) != set(expected_names):
        missing = set(expected_names) - set(submission.video_name)
        extra = set(submission.video_name) - set(expected_names)
        raise ValueError(f"Video list mismatch: {len(missing)} missing, {len(extra)} extra")
    if not submission.label.isin(labels).all():
        raise ValueError("Label outside the 100 classes")
