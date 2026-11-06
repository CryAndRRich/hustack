from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from common.experiments import run_experiments
from cv.catalog import class_names
from cv.config import Config, find_data_root, resolve_output_dir
from cv.main import PROB_COLS, confidence_summary, log_validation, run, write_submission
from cv.metrics import summarize

SHARED_KEYS = ("seed", "val_size", "split", "debug", "data_root")


def run_many(base: Config, experiments: dict[str, dict]) -> dict:
    results, root, log = run_experiments(base, experiments, SHARED_KEYS, run, resolve_output_dir, "the val/test set")
    return ensemble(results, root, base, log)


def mean_finite(arrays: list[np.ndarray]) -> tuple[np.ndarray, int]:
    stack = np.stack(arrays)
    valid = np.isfinite(stack).all(2)
    if not valid.any(0).all():
        raise ValueError(f"{int((~valid.any(0)).sum())} rows with no finite probabilities in any run")
    total = np.where(valid[..., None], stack, 0).sum(0)
    return (total / valid.sum(0)[:, None]).astype(np.float32), int((~valid.all(0)).sum())


def ensemble(results: dict[str, dict], root: Path, base: Config, log=print) -> dict:
    split = base.split
    names = class_names(find_data_root(base.data_root))
    tests = {n: pd.read_csv(Path(r["output_dir"]) / f"{split}_probs.csv") for n, r in results.items()}
    first = next(iter(results))
    for n in results:
        if not tests[n].video_name.equals(tests[first].video_name):
            raise ValueError(f"{split} video list of {n!r} differs from {first!r}")
    test_probs, partial = mean_finite([df[PROB_COLS].to_numpy() for df in tests.values()])
    videos = tests[first].video_name.tolist()
    if partial:
        log(f"[{split}] WARNING: {partial} videos lack finite probabilities in >=1 run -> averaging the remaining runs")
    if "denoised" in tests[first]:
        log(f"[{split}] median filter (noise detected): {int(tests[first].denoised.sum())}/{len(videos)} video")

    preds = {n: df[PROB_COLS].to_numpy().argmax(1) for n, df in tests.items()}
    agree = {f"{a}~{b}": round(float((preds[a] == preds[b]).mean()), 3)
             for i, a in enumerate(preds) for b in list(preds)[i + 1:]}
    for n, df in tests.items():
        log(f"[{split}] {n}: {confidence_summary(df[PROB_COLS].to_numpy())}")
    log(f"[{split}] ENSEMBLE: {confidence_summary(test_probs)} | run agreement: {agree}")

    val_files = {n: Path(r["output_dir"]) / "val_preds.csv" for n, r in results.items()}
    if all(p.exists() for p in val_files.values()):
        vals = {n: pd.read_csv(p) for n, p in val_files.items()}
        for n in results:
            if not vals[n].path.equals(vals[first].path):
                raise ValueError(f"Val set of {n!r} differs from {first!r}")
        y = vals[first].label.to_numpy()
        rows = []
        for n, df in vals.items():
            m = summarize(y, df[PROB_COLS].to_numpy())
            rows.append({"run": n, "f1": m["f1"], "acc": m["acc"], "loss": m["loss"]})
        val_probs, _ = mean_finite([df[PROB_COLS].to_numpy() for df in vals.values()])
        m = log_validation("ensemble val", y, val_probs, names, log)
        rows.append({"run": "ENSEMBLE", "f1": m["f1"], "acc": m["acc"], "loss": m["loss"]})
        noisy_files = {n: Path(r["output_dir"]) / "val_noisy_preds.csv" for n, r in results.items()}
        if all(p.exists() for p in noisy_files.values()):
            noisy = {n: pd.read_csv(p)[PROB_COLS].to_numpy() for n, p in noisy_files.items()}
            for row in rows[:-1]:
                row["f1_noisy"] = summarize(y, noisy[row["run"]])["f1"]
            rows[-1]["f1_noisy"] = summarize(y, mean_finite(list(noisy.values()))[0])["f1"]
        val_out = vals[first][["path", "label"]].copy()
        val_out[PROB_COLS] = val_probs
        val_out.to_csv(root / "ensemble_val_preds.csv", index=False)
    else:
        rows = [{"run": n, **{k: r["val"].get(k) for k in ("f1", "acc", "loss")}} for n, r in results.items()]
        log("[ensemble] no val_preds.csv (predicting from checkpoint) — val below comes from the checkpoint")
    table = pd.DataFrame(rows).set_index("run").round(4)
    log("[ensemble] validation:\n" + table.to_string())

    csv_path, zip_path = write_submission(videos, test_probs, names, root, log)
    test_out = pd.DataFrame(test_probs, columns=PROB_COLS)
    test_out.insert(0, "video_name", videos)
    test_out.to_csv(root / f"ensemble_{split}_probs.csv", index=False)
    result = {
        "val": table.reset_index().to_dict(orient="records"),
        "runs": {n: r["output_dir"] for n, r in results.items()},
        "test_agreement": agree,
        "submission_csv": str(csv_path),
        "submission_zip": str(zip_path),
    }
    (root / "ensemble_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
