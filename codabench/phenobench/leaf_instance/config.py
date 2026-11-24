from dataclasses import dataclass

from common.config import Config as Base
from common.config import roles


@dataclass
class Config(Base):
    best_fingerprints: dict = roles("plant_m2f", "leaf_eomt", "plant_zoom_m2f")
    session_hours: float = 8.0
