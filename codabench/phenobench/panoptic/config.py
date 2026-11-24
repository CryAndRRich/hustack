from dataclasses import dataclass, field

from common.config import Config as Base
from common.config import map_thresholds, roles


@dataclass
class Config(Base):
    best_fingerprints: dict = roles("plant_m2f", "plant_zoom_m2f", "crop_cls/rank0.pt", "crop_cls/rank1.pt", "sem_m2f")
    thresholds: list = field(default_factory=map_thresholds)
    sem_views: dict = field(default_factory=lambda: {"g": [1024, 1344], "t": [672, 1344]})
    weed_crop_thresholds: list = field(default_factory=lambda: [0.2, 0.35, 0.5, 0.65, 0.8])
    recover_areas: list = field(default_factory=lambda: [None, 20, 50, 100, 200])
