from __future__ import annotations

import json
import sys
import tempfile
import time
from dataclasses import asdict, replace
from pathlib import Path

from common.jobs import reset_claims, run_workers
from common.utils import Logger
from cv.catalog import assign_folds, assign_rows, scan, write_catalog
from cv.config import MARKER, NUM_CLASSES, TRACK, Config, check_experiments
from cv.data import build_cache
from cv.ensemble import aggregate

AGGREGATE_RESERVE_S = 120


def prefetch_weights(configs: list[Config], log) -> None:
    from torchvision import models

    for arch in sorted({c.arch for c in configs if c.pretrained}):
        started = time.time()
        models.get_model_weights(arch).DEFAULT.get_state_dict(progress=False)
        log(f"[weights] {arch} ready ({time.time() - started:.0f}s)")


def run(base: Config | None = None, experiments: dict[str, dict] | None = None) -> dict:
    t0 = time.time()
    base = base or Config()
    cfg = base.with_paths(TRACK, MARKER, cache_dir=base.cache_dir or str(Path(tempfile.gettempdir()) / "olpai_vn_2025_cv_cache"))
    experiments = experiments or {"main": {}}
    check_experiments(experiments)
    configs = [replace(cfg, **overrides) for overrides in experiments.values()]
    out = Path(cfg.output_dir)
    reset_claims(out / "jobs")
    log = Logger(out / "train.log", t0)
    log(f"[config] {json.dumps(asdict(cfg), ensure_ascii=False)}")
    log(f"[experiments] {experiments}")

    rows = scan(Path(cfg.data_root), cfg.negative_aspect, log)
    assign_folds(rows, cfg.folds, cfg.seed)
    positives = assign_rows(rows)
    write_catalog(rows, out / "catalog.csv")
    train_pos = [r for r in positives if r["split"] == "train"]
    for fold in range(cfg.folds):
        part = [r for r in train_pos if r["fold"] == fold]
        counts = [sum(r["label"] == k for r in part) for k in range(NUM_CLASSES)]
        log(f"[fold {fold}] {len({r['storm'] for r in part})} storms, {len(part)} images, categories 1–4: {counts}")

    cache = build_cache(Path(cfg.data_root), [f"{r['split']}/images/{r['file_name']}" for r in positives],
                        Path(cfg.cache_dir), cfg.cache_size, log)
    prefetch_weights(configs, log)

    jobs = [{"name": f"{exp}_f{fold}", "exp": exp, "exp_index": i, "fold": fold}
            for i, exp in enumerate(experiments) for fold in cfg.run_fold_list]
    spec = {"t0": t0, "deadline": t0 + cfg.time_budget_min * 60 - AGGREGATE_RESERVE_S, "output_dir": str(out),
            "cache": str(cache), "base": asdict(cfg), "experiments": experiments, "jobs": jobs}
    run_workers("cv.worker", spec, out, len(jobs), log)
    result = aggregate(out, log)
    log(f"[done] total {(time.time() - t0) / 60:.1f} min")
    return result


if __name__ == "__main__":
    run(Config().with_overrides(sys.argv[1:]))
