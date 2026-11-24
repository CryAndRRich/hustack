from dataclasses import dataclass, field

from common.config import Config as Base
from common.config import map_thresholds, roles


@dataclass
class Config(Base):
    best_fingerprints: dict = roles("plant_m2f", "leaf_eomt", "plant_zoom_m2f", "crop_cls/rank0.pt", "crop_cls/rank1.pt")
    session_target_hours: float = 9.0
    max_train_hours: float = 5.0
    thresholds: list = field(default_factory=map_thresholds)
    sem_init: list = field(default_factory=lambda: ["plant_m2f", "plant_zoom_m2f"])
    weed_scores: list = field(default_factory=lambda: [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
