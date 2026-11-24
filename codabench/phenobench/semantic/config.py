from dataclasses import dataclass, field

from common.config import Config as Base
from common.config import map_thresholds, roles


@dataclass
class Config(Base):
    best_fingerprints: dict = roles("plant_m2f", "leaf_eomt", "plant_zoom_m2f", "crop_cls/rank0.pt", "crop_cls/rank1.pt",
                                    "sem_m2f")
    session_target_hours: float = 6.5
    min_train_hours: float = 2.0
    max_train_hours: float = 2.0
    thresholds: list = field(default_factory=map_thresholds)
    crop_decision: dict = field(default_factory=lambda: {"crop_threshold": 0.8, "threshold": 0.9, "offset": 0.02})
    sem_train: list = field(default_factory=lambda: [
        {"name": "b", "kind": "m2f", "init": "sem_m2f", "size": 1024, "stride": 2, "zoom": [2.0, 3.0], "quick": [672, 1344]},
        {"name": "c", "kind": "eomt", "init": "leaf_eomt", "size": 768, "stride": 1, "zoom": [1.0, 3.0], "quick": [1024, 768]}])
    sem_gamma: float = 1.4
    sem_input: int = 1344
    weed_factors: list = field(default_factory=lambda: [0.5, 0.75, 1.0, 1.25, 1.5, 2.0])
