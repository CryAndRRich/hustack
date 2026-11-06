from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from nlp.config import SPLITS
from nlp.data import pool_gold, read_alignment, read_sentences
from nlp.decode import DEFAULT, decode, load_candidates, tune
from nlp.metric import edge_counts, score, total, validate, write_submission


def kind_of(group) -> str:
    en, vi = group
    if not en or not vi:
        return "-1"
    return f"{len(en)}{len(vi)}" if len(en) <= 2 and len(vi) <= 2 else "other"


def per_kind_recall(gold: list, pred: list) -> dict:
    hit = set(pred)
    counts, hits = Counter(), Counter()
    for g in gold:
        counts[kind_of(g)] += 1
        hits[kind_of(g)] += g in hit
    return {k: f"{hits[k]}/{counts[k]}" for k in sorted(counts)}


def aggregate(out_dir: Path, log=print, data_root: Path | None = None) -> dict:
    out_dir = Path(out_dir)
    spec = json.loads((out_dir / "run_spec.json").read_text(encoding="utf-8"))
    base = spec["base"]
    data_root = Path(data_root or base["data_root"])
    groups = read_alignment(data_root)
    texts = {s: read_sentences(data_root, s) for s in ("train",) + SPLITS}

    pools, report = [], {"pools": {}}
    for pool in spec["pools"]:
        path = out_dir / "pools" / pool["name"] / "candidates.csv"
        if not path.exists():
            log(f"[align] {pool['name']}: NO candidates (skipped / error)")
            continue
        en_text, vi_text = texts[pool["split"]]
        item = {**pool, "cands": load_candidates(path, en_text, vi_text, pool["en"])}
        if pool["split"] == "train":
            item["gold"] = pool_gold(pool, groups)
            present = {(c.en, c.vi) for c in item["cands"]}
            report["pools"][pool["name"]] = {"candidate_recall": per_kind_recall(
                [g for g in item["gold"] if g[0] and g[1]], [g for g in item["gold"] if g in present])}
        pools.append(item)
    val = [p for p in pools if "gold" in p]
    tests = {p["split"]: p for p in pools if p["split"] in SPLITS}
    missing = [s for s in SPLITS if s not in tests]
    if missing:
        raise RuntimeError(f"missing candidates for {missing} — cannot write submission")
    for name, info in report["pools"].items():
        log(f"[align] {name}: candidate recall (gold groups present in table) {info['candidate_recall']}")

    has_rr = any(c.z_rr != 0 for p in pools for c in p["cands"])
    variants = {"full": {}, "bi": {"a": 0.0}, "rr": {"a": 1.0}} if has_rr else {"bi": {"a": 0.0}}
    fixed = base.get("decoder") or {}
    params = {}
    for name, extra in variants.items():
        if val and not fixed:
            params[name], result = tune(val, extra)
        else:
            params[name] = {**DEFAULT, **extra, **fixed}
            result = total([edge_counts(p["gold"], decode(p["cands"], params[name], p["en"], p["vi"])) for p in val]) \
                if val else {"f1": float("nan")}
        log(f"[align] variant {name}: val edge F1 {result['f1']:.4f} | params {params[name]}")
        report.setdefault("variants", {})[name] = {"params": params[name], "val_edge_f1": result["f1"]}
    best = max(variants, key=lambda v: report["variants"][v]["val_edge_f1"] if val else 0)
    report["best_variant"] = best

    if val:
        edge, edge_all = [], []
        for p in val:
            pred = decode(p["cands"], params[best], p["en"], p["vi"])
            s = score(p["gold"], pred)
            edge.append(edge_counts(p["gold"], pred))
            edge_all.append(edge_counts(p["gold"], pred, include_unaligned=True))
            report["pools"][p["name"]].update(edge_f1=s["edge"]["f1"], edge_all_f1=s["edge_all"]["f1"],
                                              group_recall=per_kind_recall(p["gold"], pred))
            log(f"[val {p['name']}] edge F1 {s['edge']['f1']:.4f} | F1 incl. -1 {s['edge_all']['f1']:.4f} | "
                f"exact group recall by kind {report['pools'][p['name']]['group_recall']}")
        report["val"] = {"edge": total(edge), "edge_all": total(edge_all)}
        log(f"[val TOTAL] variant {best}: edge F1 {report['val']['edge']['f1']:.4f} "
            f"(P {report['val']['edge']['precision']:.4f} R {report['val']['edge']['recall']:.4f}) | "
            f"F1 incl. -1 {report['val']['edge_all']['f1']:.4f}")

    report["submission"] = {}
    for split in SPLITS:
        p = tests[split]
        prefix = split.split("_")[0]
        for name in variants:
            pred = decode(p["cands"], params[name], p["en"], p["vi"])
            validate(pred, p["en"], p["vi"])
            path = out_dir / (f"{prefix}_nlp.csv" if name == best else f"variants/{prefix}_nlp_{name}.csv")
            path.parent.mkdir(parents=True, exist_ok=True)
            write_submission(path, pred)
            kinds = dict(sorted(Counter(kind_of(g) for g in pred).items()))
            report["submission"][str(path.relative_to(out_dir))] = kinds
            log(f"[submission] {path.relative_to(out_dir)}: {len(pred)} rows, group kinds {kinds}")
    (out_dir / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--data-root", type=Path, default=None, help="default: data_root in run_spec.json")
    args = parser.parse_args()
    aggregate(args.output_dir, data_root=args.data_root)
