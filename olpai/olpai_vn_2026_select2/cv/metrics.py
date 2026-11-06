from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score


def macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="macro"))


def safe_auc(y_true, probs) -> float:
    try:
        return float(roc_auc_score(y_true, probs))
    except ValueError:
        return float("nan")


def binary_log_loss(y_true, probs) -> float:
    y = np.asarray(y_true, dtype=np.float64)
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def best_threshold(y_true, probs):
    grid = np.round(np.arange(0.1, 0.9 + 1e-9, 0.01), 4)
    probs = np.asarray(probs)
    scores = np.array([macro_f1(y_true, (probs >= t).astype(int)) for t in grid])
    ties = np.flatnonzero(np.isclose(scores, scores.max()))
    best = ties[np.argmin(np.abs(grid[ties] - 0.5))]
    return float(grid[best]), float(scores[best])


def validate_submission(submission: pd.DataFrame, expected_names: list[str]) -> None:
    if list(submission.columns) != ["file_name", "category_id"]:
        raise ValueError(f"Wrong column names/order: {list(submission.columns)}")
    if submission.file_name.duplicated().any():
        raise ValueError("Duplicate file_name")
    if set(submission.file_name) != set(expected_names):
        missing = set(expected_names) - set(submission.file_name)
        extra = set(submission.file_name) - set(expected_names)
        raise ValueError(f"Image list mismatch: {len(missing)} missing, {len(extra)} extra")
    if not submission.category_id.isin([0, 1]).all():
        raise ValueError("category_id has values outside {0, 1}")
