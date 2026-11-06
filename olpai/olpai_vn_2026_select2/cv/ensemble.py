from __future__ import annotations

import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from common.experiments import run_experiments
from cv.config import Config, resolve_output_dir
from cv.main import run
from cv.metrics import best_threshold, macro_f1, safe_auc, validate_submission

SHARED_KEYS = ("seed", "val_size", "split", "debug")


def run_many(base: Config, experiments: dict[str, dict]) -> dict:
    results, root, log = run_experiments(base, experiments, SHARED_KEYS, run, resolve_output_dir, "val/test set")
    return ensemble(results, root, base, log)


def ensemble(results: dict[str, dict], root: Path, base: Config, log=print) -> dict:
    split = base.split
    vals = {n: pd.read_csv(Path(r["output_dir"]) / "val_preds.csv") for n, r in results.items()}
    tests = {n: pd.read_csv(Path(r["output_dir"]) / f"{split}_probs.csv") for n, r in results.items()}
    first = next(iter(results))
    for n in results:
        if not vals[n].file_name.equals(vals[first].file_name):
            raise ValueError(f"Val set of {n!r} differs from {first!r}")
        if not tests[n].file_name.equals(tests[first].file_name):
            raise ValueError(f"{split} image list of {n!r} differs from {first!r}")

    y = vals[first].label.to_numpy()
    rows = []
    for n, df in vals.items():
        p = df.prob_fake.to_numpy()
        rows.append({"run": n, "f1@0.5": macro_f1(y, (p >= 0.5).astype(int)),
                     "f1@tuned": best_threshold(y, p)[1], "auc": safe_auc(y, p)})

    val_probs = np.mean([df.prob_fake.to_numpy() for df in vals.values()], axis=0)
    test_probs = np.mean([df.prob_fake.to_numpy() for df in tests.values()], axis=0)
    tuned, f1_tuned = best_threshold(y, val_probs)
    threshold = tuned if base.tune_threshold else 0.5
    summary = {"run": "ENSEMBLE", "f1@0.5": macro_f1(y, (val_probs >= 0.5).astype(int)),
               "f1@tuned": f1_tuned, "auc": safe_auc(y, val_probs)}
    table = pd.DataFrame(rows + [summary]).set_index("run").round(4)
    log("[ensemble] validation Macro F1:\n" + table.to_string())
    log(f"[ensemble] threshold {threshold:.2f} (tuned {tuned:.2f})")

    names = tests[first].file_name.tolist()
    submission = pd.DataFrame({"file_name": names, "category_id": (test_probs >= threshold).astype(int)})
    validate_submission(submission, names)
    csv_path, zip_path = root / "submission.csv", root / "submission.zip"
    submission.to_csv(csv_path, index=False)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(csv_path, arcname="submission.csv")
    if split == "private_test":
        shutil.copyfile(zip_path, root / "private_submission.zip")
    log(f"[submit] {zip_path} ({len(results)} models) | label counts: "
        f"{submission.category_id.value_counts().to_dict()}")

    pd.DataFrame({"file_name": vals[first].file_name, "label": y, "prob_fake": val_probs,
                  **{n: df.prob_fake for n, df in vals.items()}}).to_csv(root / "ensemble_val_preds.csv", index=False)
    pd.DataFrame({"file_name": names, "prob_fake": test_probs,
                  **{n: df.prob_fake for n, df in tests.items()}}).to_csv(root / f"ensemble_{split}_probs.csv", index=False)
    result = {
        "val": table.reset_index().to_dict(orient="records"),
        "threshold": threshold,
        "runs": {n: r["output_dir"] for n, r in results.items()},
        "submission_csv": str(csv_path),
        "submission_zip": str(zip_path),
    }
    (root / "ensemble_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
