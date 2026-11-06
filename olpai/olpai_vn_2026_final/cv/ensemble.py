from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from common.experiments import run_experiments
from cv.config import Config, resolve_output_dir
from cv.main import PROB_COLS, log_validation, run, write_submission
from cv.metrics import summarize

SHARED_KEYS = ("seed", "val_size", "split", "debug", "split_mode", "min_copies", "versions_per_image", "group_arch")


def run_many(base: Config, experiments: dict[str, dict]) -> dict:
    results, root, log = run_experiments(base, experiments, SHARED_KEYS, run, resolve_output_dir, "val/test set")
    return ensemble(results, root, base, log)


def ensemble(results: dict[str, dict], root: Path, base: Config, log=print) -> dict:
    split = base.split
    tests = {n: pd.read_csv(Path(r["output_dir"]) / f"{split}_probs.csv") for n, r in results.items()}
    first = next(iter(results))
    for n in results:
        if not tests[n].file_name.equals(tests[first].file_name):
            raise ValueError(f"{split} image list of {n!r} differs from {first!r}")
    test_probs = np.mean([df[PROB_COLS].to_numpy() for df in tests.values()], axis=0)
    names = tests[first].file_name.tolist()

    val_files = {n: Path(r["output_dir"]) / "val_preds.csv" for n, r in results.items()}
    per_class = None
    if all(p.exists() for p in val_files.values()):
        vals = {n: pd.read_csv(p) for n, p in val_files.items()}
        for n in results:
            if not vals[n].file_name.equals(vals[first].file_name):
                raise ValueError(f"Val set of {n!r} differs from {first!r}")
        y = vals[first].label.to_numpy()
        rows = []
        for n, df in vals.items():
            m = summarize(y, df[PROB_COLS].to_numpy())
            rows.append({"run": n, "f1": m["f1"], "acc": m["acc"], "loss": m["loss"]})
        val_probs = np.mean([df[PROB_COLS].to_numpy() for df in vals.values()], axis=0)
        m = log_validation("ensemble val", y, val_probs, log)
        rows.append({"run": "ENSEMBLE", "f1": m["f1"], "acc": m["acc"], "loss": m["loss"]})
        per_class = m["per_class_f1"]
        val_out = vals[first][["file_name", "label", "group"]].copy()
        val_out[PROB_COLS] = val_probs
        val_out.to_csv(root / "ensemble_val_preds.csv", index=False)
    else:
        rows = [{"run": n, **{k: r["val"].get(k) for k in ("f1", "acc", "loss")}} for n, r in results.items()]
        log("[ensemble] no val_preds.csv (checkpoint predictions) — val below comes from checkpoint")
    table = pd.DataFrame(rows).set_index("run").round(4)
    log("[ensemble] validation:\n" + table.to_string())

    csv_path, zip_path = write_submission(names, test_probs, root, log)
    test_out = pd.DataFrame(test_probs, columns=PROB_COLS)
    test_out.insert(0, "file_name", names)
    test_out.to_csv(root / f"ensemble_{split}_probs.csv", index=False)
    result = {
        "val": table.reset_index().to_dict(orient="records"),
        "per_class_f1": per_class,
        "runs": {n: r["output_dir"] for n, r in results.items()},
        "submission_csv": str(csv_path),
        "submission_zip": str(zip_path),
    }
    (root / "ensemble_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
