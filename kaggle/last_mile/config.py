from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Config:
    code_dir: str = "/tmp/code"
    work: str = "/tmp/work"
    seed: int = 2027
    folds: int = 5
    inner_folds: int = 5
    spline_knots: int = 8
    lambdas: tuple = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)
    bootstrap: int = 2000
    margin_se: float = 2.0
    reference_oof_mae: float = 7.1689
    smoke_rows: int = 0
    lo: float = 1.0
    hi: float = 365.0


def smoke(cfg):
    return replace(cfg, folds=2, inner_folds=2, spline_knots=4, lambdas=(1e-3, 1e-1), bootstrap=100, smoke_rows=1200)
