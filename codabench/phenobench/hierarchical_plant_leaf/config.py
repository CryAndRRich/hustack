from dataclasses import dataclass, field

from common.config import Config as Base


@dataclass
class Config(Base):
    leaf_gates: list = field(default_factory=lambda: [0.9, 0.94, 0.96])
