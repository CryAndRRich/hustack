from dataclasses import dataclass

from common.config import BaseConfig


@dataclass
class Config(BaseConfig):
    radius: float = 100.0
    blur_sigma: float = 1.0
    coarse_step: float = 1.0
    fine_half_width: float = 1.5
    fine_step: float = 0.1
    sift_ratio: float = 0.75
    ransac_px: float = 3.0
    min_inliers: int = 8
    disagree_deg: float = 1.0
    n_synthetic: int = 200
    jpeg_quality: int = 75
    label_max: float = 180.0
    local_data: str = "data/cv"
