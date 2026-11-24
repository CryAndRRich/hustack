from dataclasses import dataclass, replace


def yolo_job(name, weights, imgsz, batch, fraction=1.0, epochs=0):
    return {"name": name, "weights": weights, "imgsz": imgsz, "batch": batch, "fraction": fraction, "epochs": epochs}


@dataclass
class Config:
    yolo_jobs: tuple = (yolo_job("y", "yolo11l.pt", 800, 8),)
    mosaic: float = 1.0
    lr0: float = 0.01
    yolo_workers: int = 4
    dfine_epochs: int = 1000
    dfine_plain_epochs: int = 3
    dfine_batch: int = 6
    dfine_lr: float = 0.000125
    dfine_workers: int = 3
    train_hours: float = 10.2
    dfine_margin_s: int = 300
    kill_grace_hours: float = 0.3
    v1_imgsz: int = 640
    v1_sha: str = "5e041ac9749e454d"
    seed: int = 0
    nms_iou: float = 0.5
    multi_label: bool = True
    wbf_iou: float = 0.55
    max_subset: int = 3
    top_arms: int = 4
    conf: float = 0.001
    max_det: int = 100
    eval_limit: int = 0
    latency_images: int = 300
    warmup: int = 20
    fast_ms: float = 25.0
    slow_ms: float = 250.0
    heartbeat_s: int = 600
    smoke: bool = False
    data_dir: str = "/tmp/data"
    run_dir: str = "/tmp/runs"
    dfine_dir: str = "/tmp/dfine"
    work_dir: str = "/tmp/work"
    code_dir: str = "/tmp/code"
    submission_name: str = "submission.csv"


def smoke_config(cfg):
    jobs = tuple({**j, "imgsz": 320, "batch": 16, "fraction": 0.02, "epochs": 1} for j in cfg.yolo_jobs[:1])
    return replace(cfg, yolo_jobs=jobs, dfine_epochs=2, dfine_plain_epochs=1, train_hours=0.4, eval_limit=64,
                   latency_images=20, warmup=3, top_arms=2, heartbeat_s=60, smoke=True, run_dir=cfg.run_dir + "/smoke")
