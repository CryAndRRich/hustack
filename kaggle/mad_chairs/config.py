from dataclasses import dataclass


@dataclass
class Config:
    code_dir: str = "/tmp/code"
    work: str = "/tmp/work"
    strategy_name: str = "predictblock"
    runs: int = 4
    log_every_s: float = 120.0
