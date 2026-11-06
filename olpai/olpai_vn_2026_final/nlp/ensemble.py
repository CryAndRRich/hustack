from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from common.experiments import run_experiments
from nlp.config import Config, resolve_output_dir
from nlp.data import load_examples, split_path, test_split_name
from nlp.main import DEV_PREDICTIONS, read_predictions, run, write_submission
from nlp.metric import evaluate, format_scores

SHARED_KEYS = ("split", "test_file", "debug")


def run_many(base: Config, experiments: dict[str, dict]) -> dict:
    results, root, log = run_experiments(base, experiments, SHARED_KEYS, run, resolve_output_dir, "test set")
    return ensemble(results, root, base, log)


def vote(runs: list[dict], min_votes: int) -> dict:
    out = {}
    for key in runs[0]:
        counts, order = Counter(), {}
        for preds in runs:
            for t in dict.fromkeys(preds.get(key, [])):
                counts[t] += 1
                order.setdefault(t, len(order))
        out[key] = sorted((t for t, c in counts.items() if c >= min_votes), key=order.get)
    return out


def ensemble(results: dict[str, dict], root: Path, base: Config, log=print) -> dict:
    run_config = next(iter(results.values()))["config"]
    data_root = Path(run_config["data_root"])
    test_path = Path(run_config["test_file"]) if run_config["test_file"] else split_path(data_root, test_split_name(base.split))
    test_by_key = {(ex.domain, ex.index): ex for ex in load_examples(test_path)}
    dev_by_key = {(ex.domain, ex.index): ex for ex in load_examples(split_path(data_root, "Dev"))}

    names = list(results)
    tests = [read_predictions(Path(r["output_dir"]) / f"{base.split}_predictions.jsonl") for r in results.values()]
    dev_files = [Path(r["output_dir"]) / DEV_PREDICTIONS for r in results.values()]
    test = [test_by_key[k] for k in tests[0]]

    table = []
    majority = len(names) // 2 + 1
    best_votes = majority
    if all(p.exists() for p in dev_files):
        devs = [read_predictions(p) for p in dev_files]
        dev = [dev_by_key[k] for k in devs[0]]
        for name, preds in zip(names, devs):
            table.append({"run": name, "dev_f1": round(evaluate(dev, preds)["avg_f1"], 4)})
        scores = {k: evaluate(dev, vote(devs, k)) for k in range(1, len(names) + 1)}
        for k, s in scores.items():
            table.append({"run": f"VOTE>={k}", "dev_f1": round(s["avg_f1"], 4)})
        best_votes = max(scores, key=lambda k: (scores[k]["avg_f1"], k == majority))
        log("[ensemble] F1 Dev:\n" + "\n".join(f"  {r['run']:<12} {r['dev_f1']:.4f}" for r in table))
        log(f"[ensemble] chose min_votes={best_votes}:\n" + format_scores(scores[best_votes]))
    else:
        log(f"[ensemble] no Dev predictions (train_on_dev?) -> majority min_votes={majority}")

    preds = vote(tests, best_votes)
    json_path, zip_path = write_submission(test, preds, root, log)
    result = {"val": table, "min_votes": best_votes, "runs": {n: r["output_dir"] for n, r in results.items()},
              "submission_json": str(json_path), "submission_zip": str(zip_path)}
    (root / "ensemble_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
