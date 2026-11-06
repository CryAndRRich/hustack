from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import asdict, replace
from pathlib import Path

from common.distributed import RESULT_FILE
from common.paths import PROJECT_ROOT
from common.utils import Logger
from nlp.config import Config, find_data_root, resolve_output_dir
from nlp.data import split_pairs
from nlp.main import SUBMISSION_CSV, SUBMISSION_ZIP, load_data
from nlp.tokenizer import train_spm

SHARED_KEYS = ("seed", "val_size", "dedupe_pairs", "split", "debug", "data_root", "vi_tokenizer", "vi_vocab", "vi_spm")


def train_shared_vi(base: Config, root: Path, log) -> Path:
    cfg = base.debug_variant() if base.debug else base
    cfg = replace(cfg, data_root=str(find_data_root(base.data_root)))
    pairs, _ = load_data(cfg)
    train, _ = split_pairs(pairs, cfg.val_size, cfg.dedupe_pairs, cfg.seed, log)
    path = train_spm([vi for _, vi in train], root / "spm_vi_shared", cfg.vi_tokenizer, cfg.vi_vocab,
                     byte_fallback=False)
    log(f"[runner] shared vi tokenizer: {path}")
    return path


def _pump(process: subprocess.Popen, name: str) -> None:
    for line in process.stdout:
        print(f"[{name}] {line}", end="", flush=True)


def launch_parallel(jobs: list[tuple[str, Path]], n_gpu: int, log) -> list[str]:
    slots = list(range(max(1, n_gpu)))
    pending, running, failed = list(jobs), {}, []
    while pending or running:
        while pending and slots:
            gpu = slots.pop(0)
            name, config_path = pending.pop(0)
            env = dict(os.environ, PYTHONUNBUFFERED="1",
                       PYTHONPATH=os.pathsep.join(p for p in [str(PROJECT_ROOT), os.environ.get("PYTHONPATH", "")] if p))
            if n_gpu > 0:
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            process = subprocess.Popen([sys.executable, "-m", "nlp.main", "--config", str(config_path)],
                                       cwd=PROJECT_ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, bufsize=1)
            thread = threading.Thread(target=_pump, args=(process, name), daemon=True)
            thread.start()
            running[gpu] = (name, process, thread)
            log(f"[runner] {name} -> GPU {gpu if n_gpu > 0 else 'cpu'}")
        time.sleep(2)
        for gpu, (name, process, thread) in list(running.items()):
            if process.poll() is not None:
                thread.join()
                del running[gpu]
                slots.append(gpu)
                if process.returncode != 0:
                    failed.append(name)
                    log(f"[runner] {name} FAILED (exit code {process.returncode}) — see log above")
    return failed


def publish(source: Path, root: Path) -> None:
    for file in (SUBMISSION_CSV, SUBMISSION_ZIP):
        shutil.copy(source / file, root / file)


def run_many(base: Config, experiments: dict[str, dict]) -> dict:
    import torch

    if not experiments:
        raise ValueError("experiments is empty")
    for name, overrides in experiments.items():
        clash = [k for k in overrides if k in SHARED_KEYS]
        if clash:
            raise ValueError(f"Experiment {name!r} must not change {clash}: runs must share the val/test set")

    root = resolve_output_dir(base.output_dir)
    if base.debug:
        root = root / "debug"
    root.mkdir(parents=True, exist_ok=True)
    log = Logger(root / "runner.log")

    checkpoint_mode = bool(base.checkpoint) or any(o.get("checkpoint") for o in experiments.values())
    if not checkpoint_mode and base.vi_spm is None and len(experiments) > 1:
        base = replace(base, vi_spm=str(train_shared_vi(base, root, log)))

    jobs = []
    for name, overrides in experiments.items():
        cfg = replace(base, output_dir=str(root / "runs" / name), **overrides)
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
        config_path = Path(cfg.output_dir) / "run_config.json"
        config_path.write_text(json.dumps(asdict(cfg), ensure_ascii=False, indent=2), encoding="utf-8")
        (Path(cfg.output_dir) / RESULT_FILE).unlink(missing_ok=True)
        jobs.append((name, config_path))
    failed = launch_parallel(jobs, torch.cuda.device_count(), log)

    results = {}
    for name, _ in jobs:
        path = root / "runs" / name / RESULT_FILE
        if name not in failed and path.exists():
            results[name] = json.loads(path.read_text(encoding="utf-8"))
    if not results:
        raise RuntimeError(f"All experiments failed: {failed}")

    table = [{"run": n, "val_bleu": r["val"].get("bleu"), "chosen": r["val"].get("chosen"),
              "memory_hits": r["memory_hits"]} for n, r in results.items()]
    for row in table:
        log(f"[runner] {row['run']:<12} BLEU val = {row['val_bleu']} ({row['chosen']})")
    best = max(results, key=lambda n: results[n]["val"].get("bleu") or -1)
    best_bleu, best_dir = results[best]["val"].get("bleu") or -1, Path(results[best]["output_dir"])
    publish(best_dir, root)
    log(f"[runner] provisionally chose {best} -> {root / SUBMISSION_ZIP}")

    ensemble_result = None
    if base.ensemble and len(results) > 1:
        from nlp.ensemble import ensemble_runs

        run_dirs = {n: Path(r["config"].get("checkpoint") or r["output_dir"]) for n, r in results.items()}
        ensemble_result = ensemble_runs(base, run_dirs, root / "ensemble", log)
        weights = {n: round(w, 3) for n, w in ensemble_result["weights"].items()}
        chosen = f"beam={ensemble_result['beam']} len_penalty={ensemble_result['len_penalty']} weights={weights}"
        table.append({"run": "ENSEMBLE", "val_bleu": ensemble_result["bleu"], "chosen": chosen,
                      "memory_hits": ensemble_result["memory_hits"]})
        log(f"[runner] ENSEMBLE     BLEU val = {ensemble_result['bleu']:.2f} ({chosen})")
        if ensemble_result["bleu"] > best_bleu:
            best, best_dir = "ENSEMBLE", root / "ensemble"

    publish(best_dir, root)
    log(f"[runner] chose {best} -> {root / SUBMISSION_ZIP}" + (f" | failed: {failed}" if failed else ""))
    result = {"val": table, "best": best, "failed": failed, "ensemble": ensemble_result,
              "submission_csv": str(root / SUBMISSION_CSV), "submission_zip": str(root / SUBMISSION_ZIP)}
    (root / "runner_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
