from dataclasses import dataclass

from common.config import BaseConfig


@dataclass
class Config(BaseConfig):
    features: tuple = ("charm", "passion", "filipino", "smarts", "gender", "fashion")
    label: str = "label"
    id_col: str = "id"
    n_test: int = 999_900
    local_data: str = "data/ml"
    max_positive_share: float = 1e-4
