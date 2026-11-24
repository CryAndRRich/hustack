from dataclasses import dataclass, field

from common.config import Config as Base
from common.config import roles


@dataclass
class Config(Base):
    best_fingerprints: dict = roles("plant_m2f", "leaf_eomt", "plant_zoom_m2f", "crop_cls/rank0.pt", "crop_cls/rank1.pt",
                                    "sem_m2f")
    min_score: float = 0.05
    leaf_plant_gate: float = 0.5
    yolo_model: str = "yolo11l.pt"
    yolo_imgsz: int = 1024
    yolo_batch: int = 8
    yolo_workers: int = 2
    yolo_hours: float = 2.5
    smoke_yolo_hours: float = 0.01
    smoke_train_images: int = 16
    yolo_conf: float = 0.01
    yolo_max_det: int = 300
    yolo_views: list = field(default_factory=lambda: ["1024", "1024f", "1280", "1280f"])
    yolo_tta: dict = field(default_factory=lambda: {"a": ["1024"], "b": ["1024", "1024f"],
                                                    "c": ["1024", "1024f", "1280", "1280f"]})
    tta_iou: float = 0.55
    wbf_iou: list = field(default_factory=lambda: [0.5, 0.6])
    wbf_mask_weights: list = field(default_factory=lambda: [0.5, 1.0, 2.0])
    sem_view: list = field(default_factory=lambda: [1024, 1344])
    check_images: int = 30
    final_check_images: int = 200
    final_check_minutes: int = 20
