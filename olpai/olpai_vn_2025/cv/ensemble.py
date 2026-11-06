from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from cv.catalog import read_catalog
from cv.config import NUM_CLASSES, SPLITS
from cv.metrics import argmax, format_summary, positive_summary, validate_submission, write_submission
from cv.smoothing import class_prior, smooth, transition_matrix, tune

PROB_COLS = [f"p{k}" for k in range(NUM_CLASSES)]


def read_probs(path: Path) -> dict[str, list[float]]:
    out = {}
    with open(path, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            p = [float(row[c]) for c in PROB_COLS]
            if all(math.isfinite(v) for v in p):
                out[row["file_name"]] = p
    return out


def mean_probs(parts: list[dict[str, list[float]]]) -> dict[str, list[float]]:
    sums, counts = {}, defaultdict(int)
    for part in parts:
        for name, p in part.items():
            sums[name] = [a + b for a, b in zip(sums.get(name, [0.0] * NUM_CLASSES), p)]
            counts[name] += 1
    return {name: [v / counts[name] for v in s] for name, s in sums.items()}


def write_probs(path: Path, rows: list[dict], columns: dict[str, dict[str, list[float]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["file_name", "split", "storm", "time", "label"]
                        + [f"{prefix}_{c}" for prefix in columns for c in PROB_COLS])
        for r in rows:
            values = []
            for probs in columns.values():
                p = probs.get(r["file_name"])
                values += [f"{v:.6f}" for v in p] if p else [""] * NUM_CLASSES
            writer.writerow([r["file_name"], r["split"], r["storm"], r["time"], r["label"]] + values)


def by_year(rows: list[dict], probs: dict[str, list[float]]) -> dict[str, float]:
    hits = defaultdict(list)
    for r in rows:
        hits["20" + r["storm"][:2]].append(argmax(probs[r["file_name"]]) == r["label"])
    return {year: round(sum(v) / len(v), 4) for year, v in sorted(hits.items())}


def aggregate(out_dir: Path, log=print, exps: list[str] | None = None, smooth_on: bool | None = None) -> dict:
    out_dir = Path(out_dir)
    spec = json.loads((out_dir / "run_spec.json").read_text(encoding="utf-8"))
    base = spec["base"]
    smooth_on = base["smooth"] if smooth_on is None else smooth_on
    exps = exps or list(spec["experiments"])
    rows = read_catalog(out_dir / "catalog.csv")
    train_pos = [r for r in rows if r["split"] == "train" and not r["negative"]]
    test_pos = [r for r in rows if r["split"] != "train" and not r["negative"]]

    oof, test, report = {}, {}, {"jobs": {}, "experiments": {}}
    for exp in exps:
        oof_parts, test_parts = [], []
        for job in (j for j in spec["jobs"] if j["exp"] == exp):
            job_dir = out_dir / "jobs" / job["name"]
            result_path = job_dir / "result.json"
            status = json.loads(result_path.read_text())["status"] if result_path.exists() else "missing"
            report["jobs"][job["name"]] = status
            if status == "done":
                oof_parts.append(read_probs(job_dir / "oof.csv"))
                test_parts.append(read_probs(job_dir / "test.csv"))
        if not test_parts:
            log(f"[ensemble] {exp}: no finished jobs — skipped")
            continue
        oof[exp] = {k: v for part in oof_parts for k, v in part.items()}
        test[exp] = mean_probs(test_parts)
        covered = [r for r in train_pos if r["file_name"] in oof[exp]]
        s = positive_summary([r["label"] for r in covered], [oof[exp][r["file_name"]] for r in covered])
        report["experiments"][exp] = {"folds_done": len(test_parts), "oof": s}
        log(format_summary(f"OOF {exp} ({len(test_parts)} fold)", s))
    if not test:
        raise RuntimeError("no experiment has a finished job")
    log(f"[ensemble] job status: {report['jobs']}")

    ens_oof = mean_probs(list(oof.values()))
    ens_test = mean_probs(list(test.values()))
    missing = [r["file_name"] for r in test_pos if r["file_name"] not in ens_test]
    if missing:
        raise RuntimeError(f"{len(missing)} storm test images without finite probabilities: {missing[:5]}")
    covered = [r for r in train_pos if r["file_name"] in ens_oof]
    labels = [r["label"] for r in covered]
    raw = positive_summary(labels, [ens_oof[r["file_name"]] for r in covered])
    report["oof_ensemble"] = raw
    log(format_summary(f"OOF ENSEMBLE {'+'.join(oof)}", raw) + f" | by year {by_year(covered, ens_oof)}")

    final_test, smooth_oof = ens_test, {}
    if smooth_on:
        keys_all = [(r["storm"], r["time"]) for r in train_pos]
        transition = transition_matrix(keys_all, [r["label"] for r in train_pos], NUM_CLASSES)
        prior = class_prior([r["label"] for r in train_pos], NUM_CLASSES)
        max_gap = base["smooth_max_gap_h"] // 3
        keys = [(r["storm"], r["time"]) for r in covered]
        probs = [ens_oof[r["file_name"]] for r in covered]
        best, table = tune(probs, keys, labels, transition, prior, base["smooth_alphas"], base["smooth_mixes"],
                           base["smooth_gammas"], max_gap)
        smoothed = smooth(probs, keys, transition, prior, best["alpha"], best["mix"], max_gap, best["gamma"])
        smooth_oof = {r["file_name"]: p for r, p in zip(covered, smoothed)}
        s = positive_summary(labels, smoothed)
        report["smoothing"] = {"alpha": best["alpha"], "mix": best["mix"], "gamma": best["gamma"], "oof": s, "grid": table,
                               "transition": [[round(v, 4) for v in row] for row in transition]}
        log(f"[smoothing] 3-hour category transition matrix {report['smoothing']['transition']} | prior "
            f"{[round(v, 3) for v in prior]}")
        top = sorted(table, key=lambda r: -r["acc"])[:8]
        log(f"[smoothing] {len(table)} combos, best: "
            + ", ".join(f"a{r['alpha']}/m{r['mix']}/g{r['gamma']}={r['acc']:.4f}" for r in top))
        log(format_summary(f"OOF SMOOTHED alpha={best['alpha']} mix={best['mix']} gamma={best['gamma']}", s)
            + f" | by year {by_year(covered, smooth_oof)} | unsmoothed acc {raw['acc']:.4f}")
        groups = [test_pos] if base["smooth_join_splits"] else [[r for r in test_pos if r["split"] == s] for s in SPLITS]
        final_test = {}
        for group in groups:
            post = smooth([ens_test[r["file_name"]] for r in group], [(r["storm"], r["time"]) for r in group],
                          transition, prior, best["alpha"], best["mix"], max_gap, best["gamma"])
            final_test.update({r["file_name"]: p for r, p in zip(group, post)})

    variants = {"": final_test} | ({"raw": ens_test} if smooth_on else {})
    report["submission"] = {}
    for split in SPLITS:
        split_rows = sorted((r for r in rows if r["split"] == split), key=lambda r: r["file_name"])
        names = [r["file_name"] for r in split_rows]
        for tag, probs in variants.items():
            pred = [-1 if r["negative"] else argmax(probs[r["file_name"]]) for r in split_rows]
            path = out_dir / (f"variants/{split.split('_')[0]}_cv_{tag}.csv" if tag else f"{split.split('_')[0]}_cv.csv")
            path.parent.mkdir(parents=True, exist_ok=True)
            write_submission(path, names, pred)
            counts = validate_submission(path, names)
            report["submission"][str(path.relative_to(out_dir))] = counts
            log(f"[submission] {path.relative_to(out_dir)}: {counts}")
        if smooth_on:
            changed = sum(argmax(final_test[r["file_name"]]) != argmax(ens_test[r["file_name"]])
                          for r in split_rows if not r["negative"])
            log(f"[submission] {split}: smoothing changed category on {changed}/{sum(not r['negative'] for r in split_rows)} storm images")

    write_probs(out_dir / "probs" / "oof.csv", covered,
                {**{f"oof_{e}": oof[e] for e in oof}, "ens": ens_oof, **({"smooth": smooth_oof} if smooth_on else {})})
    write_probs(out_dir / "probs" / "test.csv", test_pos,
                {**{f"test_{e}": test[e] for e in test}, "ens": ens_test, **({"smooth": final_test} if smooth_on else {})})
    (out_dir / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--exps", default=None, help="comma-separated experiments (default: all)")
    parser.add_argument("--no-smooth", action="store_true")
    args = parser.parse_args()
    aggregate(args.output_dir, exps=args.exps.split(",") if args.exps else None,
              smooth_on=False if args.no_smooth else None)
