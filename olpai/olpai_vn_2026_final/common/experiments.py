from __future__ import annotations

from dataclasses import replace

from common.utils import Logger


def run_experiments(base, experiments: dict[str, dict], shared_keys: tuple, run, output_root, scope: str):
    if not experiments:
        raise ValueError("experiments is empty")
    for name, overrides in experiments.items():
        clash = [k for k in overrides if k in shared_keys]
        if clash:
            raise ValueError(f"Experiment {name!r} must not change {clash}: runs must share {scope}")
    base_root = output_root(base.output_dir)
    root = base_root / "debug" if base.debug else base_root
    root.mkdir(parents=True, exist_ok=True)
    log = Logger(root / "ensemble.log")
    results = {}
    for i, (name, overrides) in enumerate(experiments.items(), 1):
        log(f"[ensemble] ===== {i}/{len(experiments)} {name}: {overrides}")
        results[name] = run(replace(base, output_dir=str(base_root / "runs" / name), **overrides))
    return results, root, log
