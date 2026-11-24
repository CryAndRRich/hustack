import json
import shutil
import time
from pathlib import Path

import torch

import dfine
from config import Config, smoke_config
from detector import Yolo
from evaluate import predict_test, select
from jobs import train_all
from prepare import find_kit, prepare
from runtime import run_session

SESSION_START = time.time()


def find_v1(cfg):
    root = Path("/kaggle/input")
    for pattern in ["*/yolo11m/best.pt", "*/*/yolo11m/best.pt", "*/*/*/yolo11m/best.pt", "*/*/*/*/yolo11m/best.pt"]:
        for p in sorted(root.glob(pattern)):
            sha = dfine.sha16(p)
            assert sha == cfg.v1_sha, f"{p} has sha {sha}, expected the V1 weights {cfg.v1_sha}"
            print(f"V1 weights: {p}")
            return p
    print("V1 weights not attached (dataset road-best): ensemble without the V1 member")
    return None


def load_models(cfg, status, v1, work, repo):
    models = {}
    for job in cfg.yolo_jobs:
        s = status.get(job["name"], {})
        if s.get("weights"):
            shutil.copy(s["weights"], work / f"{job['name']}.pt")
            models[job["name"]] = Yolo(s["weights"], job["imgsz"], cfg)
        else:
            print(f"{job['name']}: no weights ({s})")
    d = status.get("d", {})
    if d.get("weights"):
        models["d"] = dfine.DFine(repo, d["config"], d["weights"], cfg.max_det)
    else:
        print(f"dfine: no weights ({d})")
    if v1:
        models["v1"] = Yolo(v1, cfg.v1_imgsz, cfg)
    expected = {j["name"] for j in cfg.yolo_jobs} | {"d"}
    assert models.keys() >= expected or not cfg.smoke, f"smoke: missing models {expected - models.keys()}"
    failed = {k: v.get("code") for k, v in status.items() if v.get("code") != 0}
    assert not (cfg.smoke and failed), f"smoke: workers failed {failed}, see log_*.txt in results.zip"
    assert models, "no model trained"
    return models


def stage(cfg, kit, v1, work, submission, deadline):
    (work / "extra").mkdir(parents=True, exist_ok=True)
    data = prepare(cfg, kit)
    repo, coco_weights = dfine.setup(cfg)
    status = train_all(cfg, data, work, deadline, repo, coco_weights)
    models = load_models(cfg, status, v1, work, repo)
    chosen, report, ranking, arms = select(cfg, kit, models, work / "extra")
    files = predict_test(cfg, kit, models, arms, report, chosen, submission, work / "extra")
    del models
    torch.cuda.empty_cache()
    return {"chosen": chosen, "candidates": report, "ranking": ranking, "files": files,
            "training": {k: {x: y for x, y in v.items() if x != "epochs"} for k, v in status.items()}}


def solve(cfg, work, submission):
    assert torch.cuda.device_count() == 2, f"needs GPU T4 x2, found {torch.cuda.device_count()} GPUs"
    import albumentations
    import ultralytics
    print(f"torch {torch.__version__}, ultralytics {ultralytics.__version__}, albumentations {albumentations.__version__}, "
          f"GPU {torch.cuda.get_device_name(0)}")
    kit = find_kit()
    print(f"kit: {kit}")
    v1 = find_v1(cfg)
    smoke_cfg = smoke_config(cfg)
    smoke = stage(smoke_cfg, kit, v1, work / "smoke", work / "smoke" / "submission.csv",
                  time.time() + smoke_cfg.train_hours * 3600)
    for p in (work / "smoke").glob("*.pt"):
        p.unlink()
    print(f"smoke passed: chosen {smoke['chosen']}, training {json.dumps(smoke['training'])}")
    result = stage(cfg, kit, v1, work, submission, SESSION_START + cfg.train_hours * 3600)
    return {"smoke_chosen": smoke["chosen"], **result}


def run(cfg=None):
    run_session(cfg or Config(), solve)
