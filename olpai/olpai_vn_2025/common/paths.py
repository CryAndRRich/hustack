from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")


def find_data_root(explicit: str | None, track: str, marker: str) -> Path:
    if explicit:
        root = Path(explicit)
        if not (root / marker).is_file():
            raise FileNotFoundError(f"{marker} not found in {root}")
        return root
    env_var = f"OLPAI_{track.upper()}_DATA_ROOT"
    env = os.environ.get(env_var)
    candidates = ([Path(env)] if env else []) + [PROJECT_ROOT / "data" / track, Path.cwd() / "data" / track]
    for root in candidates:
        if (root / marker).is_file():
            return root
    if KAGGLE_INPUT.is_dir():
        parts = Path(marker).parts
        for path in sorted(KAGGLE_INPUT.rglob(parts[-1])):
            if path.parts[-len(parts):] == parts:
                return path.parents[len(parts) - 1]
    raise FileNotFoundError(f"data not found ({marker}). Set data_root or env var {env_var}.")


def resolve_output_dir(explicit: str | None, track: str) -> Path:
    if explicit:
        out = Path(explicit)
    elif KAGGLE_WORKING.is_dir():
        out = KAGGLE_WORKING
    else:
        out = PROJECT_ROOT / "outputs" / track
    out.mkdir(parents=True, exist_ok=True)
    return out
