import json
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

import dfine

YOLO_MARGIN_H = 0.1


def run_worker(cmd, cwd, gpu, log_path, kill_at, extra_env=None):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "GITHUB_ACTIONS": "1", "PYTHONUNBUFFERED": "1",
           **(extra_env or {})}
    t0 = time.time()
    with open(log_path, "w") as log:
        proc = subprocess.Popen([sys.executable, *cmd], cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
        while proc.poll() is None:
            if time.time() > kill_at:
                proc.terminate()
                try:
                    proc.wait(120)
                except subprocess.TimeoutExpired:
                    proc.kill()
                print(f"killed {log_path.name}: past its deadline")
                break
            time.sleep(10)
    return {"code": proc.wait(), "seconds": round(time.time() - t0)}


def last_line(path):
    if not path.exists():
        return ""
    with open(path, "rb") as f:
        f.seek(max(0, path.stat().st_size - 4000))
        tail = f.read().decode("utf-8", "replace").replace("\r", "\n")
    lines = [x.strip() for x in tail.splitlines() if x.strip()]
    return lines[-1][:200] if lines else ""


def yolo_chain(cfg, data, work, deadline, status):
    run_dir = Path(cfg.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    for job in cfg.yolo_jobs:
        remaining = (deadline - time.time()) / 3600 - YOLO_MARGIN_H
        if remaining <= 0:
            status[job["name"]] = {"skipped": "no time left"}
            continue
        timed = job["epochs"] == 0
        args = {"weights": job["weights"], "data": str(data / "data.yaml"), "imgsz": job["imgsz"],
                "epochs": 300 if timed else job["epochs"], "hours": remaining if timed else None,
                "batch": job["batch"], "fraction": job["fraction"], "workers": cfg.yolo_workers, "lr0": cfg.lr0,
                "mosaic": cfg.mosaic, "seed": cfg.seed, "project": str(run_dir), "name": job["name"]}
        args_path = work / f"fit_{job['name']}.json"
        args_path.write_text(json.dumps(args, indent=1))
        kill_at = deadline + cfg.kill_grace_hours * 3600
        r = run_worker([str(Path(cfg.code_dir) / "fit.py"), str(args_path)], run_dir, 0,
                       work / f"log_{job['name']}.txt", kill_at)
        out = run_dir / job["name"]
        weights = next((p for p in (out / "weights" / "best.pt", out / "weights" / "last.pt") if p.exists()), None)
        if (out / "results.csv").exists():
            shutil.copy(out / "results.csv", work / f"results_{job['name']}.csv")
        status[job["name"]] = {**r, "weights": str(weights) if weights else None, "imgsz": job["imgsz"]}


def dfine_run(cfg, data, work, deadline, repo, coco_weights, status):
    out_dir = Path(cfg.run_dir) / "dfine"
    shutil.rmtree(out_dir, ignore_errors=True)
    config = dfine.write_config(cfg, repo, data, out_dir)
    shutil.copy(config, work / "dfine_config.yml")
    args = {"repo": str(repo), "config": str(config), "tuning": str(coco_weights), "output_dir": str(out_dir),
            "seed": cfg.seed, "deadline": deadline - cfg.dfine_margin_s, "plain_epochs": cfg.dfine_plain_epochs}
    args_path = work / "fit_dfine.json"
    args_path.write_text(json.dumps(args, indent=1))
    r = run_worker([str(Path(cfg.code_dir) / "dfine_train.py"), str(args_path)], repo, 1, work / "log_dfine.txt",
                   deadline + cfg.kill_grace_hours * 3600,
                   extra_env={"TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1",
                              "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    epochs = dfine.epoch_log(out_dir)
    (work / "dfine_epochs.json").write_text(json.dumps(epochs, indent=1))
    target = work / "dfine.pt"
    try:
        used = dfine.export(out_dir, target)
    except FileNotFoundError as e:
        status["d"] = {**r, "error": str(e)}
        return
    status["d"] = {**r, "checkpoint": used, "weights": str(target), "config": str(config), "epochs": epochs}


def train_all(cfg, data, work, deadline, repo, coco_weights):
    status = {}

    def guarded(name, fn, *args):
        try:
            fn(*args)
        except Exception:
            traceback.print_exc(file=sys.stdout)
            status.setdefault(name, {})["error"] = traceback.format_exc(limit=3)

    threads = [threading.Thread(target=guarded, args=("yolo", yolo_chain, cfg, data, work, deadline, status)),
               threading.Thread(target=guarded, args=("d", dfine_run, cfg, data, work, deadline, repo, coco_weights,
                                                      status))]
    for t in threads:
        t.start()
    t0 = time.time()
    while any(t.is_alive() for t in threads):
        for t in threads:
            t.join(timeout=cfg.heartbeat_s / len(threads))
        logs = sorted(work.glob("log_*.txt"))
        print(f"[{(time.time() - t0) / 60:.0f} min] " + " | ".join(f"{p.stem[4:]}: {last_line(p)}" for p in logs))
    print("training status:", json.dumps({k: {x: y for x, y in v.items() if x != "epochs"} for k, v in status.items()}))
    return status
