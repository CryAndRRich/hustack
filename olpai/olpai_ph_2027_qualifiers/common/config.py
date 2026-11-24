from dataclasses import dataclass


@dataclass
class BaseConfig:
    n_test: int = 500
    seed: int = 2027
    local_data: str = "data"
    work_dir: str = "/tmp/work"
    code_dir: str = "/tmp/code"
    submission_name: str = "submission.csv"
