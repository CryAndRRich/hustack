from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch

from common.paths import PROJECT_ROOT
from common.utils import Logger, seed_everything


def claim(lock_dir: Path, name: str) -> bool:
    try:
        os.close(os.open(Path(lock_dir) / name, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        return True
    except FileExistsError:
        return False


def worker_setup(config_cls):
    spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    worker_id = int(sys.argv[2])
    out = Path(spec["output_dir"])
    log = Logger(out / "logs" / f"worker{worker_id}.log", spec["t0"], f"[gpu{worker_id}] ")
    cfg = config_cls(**spec["base"])
    seed_everything(cfg.seed)
    return spec, out, log, cfg, torch.device("cuda" if torch.cuda.is_available() else "cpu")


def reset_claims(work_dir: Path) -> None:
    shutil.rmtree(work_dir, ignore_errors=True)
    (work_dir / ".claims").mkdir(parents=True)


def run_workers(module: str, spec: dict, out: Path, n_jobs: int, log, extra_env: dict | None = None) -> None:
    spec_path = out / "run_spec.json"
    spec_path.write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
    use_gpu = torch.cuda.is_available()
    n = min(max(1, torch.cuda.device_count()), n_jobs)
    log(f"[worker] {n_jobs} jobs on {n} GPU | {(spec['deadline'] - time.time()) / 60:.1f} min left before deadline")
    procs = []
    for i in range(n):
        env = dict(os.environ, PYTHONUNBUFFERED="1", OMP_NUM_THREADS="2", **(extra_env or {}))
        env["PYTHONPATH"] = os.pathsep.join(p for p in [str(PROJECT_ROOT), env.get("PYTHONPATH", "")] if p)
        if use_gpu:
            env["CUDA_VISIBLE_DEVICES"] = str(i)
        procs.append(subprocess.Popen([sys.executable, "-m", module, str(spec_path), str(i)], cwd=PROJECT_ROOT,
                                      env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1))

    def pump(proc):
        for line in proc.stdout:
            print(line, end="", flush=True)

    threads = [threading.Thread(target=pump, args=(p,), daemon=True) for p in procs]
    for t in threads:
        t.start()
    codes = [p.wait() for p in procs]
    for t in threads:
        t.join()
    if any(codes):
        log(f"[worker] WARNING: exit codes {codes} — using finished jobs")
