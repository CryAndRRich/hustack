from __future__ import annotations

import os
import shutil
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")


def find_data_root(explicit: str | None, marker: str, candidates: list[Path], env_var: str) -> Path:
    if explicit:
        root = Path(explicit)
        if not (root / marker).is_file():
            raise FileNotFoundError(f"{marker} not found in {root}")
        return root

    env = os.environ.get(env_var)
    for root in ([Path(env)] if env else []) + list(candidates):
        if (root / marker).is_file():
            return root

    if KAGGLE_INPUT.is_dir():
        parts = Path(marker).parts
        for path in sorted(KAGGLE_INPUT.rglob(parts[-1])):
            if path.parts[-len(parts):] == parts:
                return path.parents[len(parts) - 1]

    raise FileNotFoundError(f"Data not found ({marker}). Set data_root or env var {env_var}.")


def resolve_output_dir(explicit: str | None, track: str) -> Path:
    if explicit:
        out = Path(explicit)
    elif KAGGLE_WORKING.is_dir():
        out = KAGGLE_WORKING
    else:
        out = PROJECT_ROOT / "outputs" / track
    out.mkdir(parents=True, exist_ok=True)
    return out


def unpack_split(data_root: Path, split: str, password: str | None, work_dir: Path) -> Path:
    data_root, work_dir = Path(data_root), Path(work_dir)
    if (data_root / split).is_dir():
        return data_root
    archive = data_root / f"{split}.zip"
    if not archive.is_file() or (work_dir / split).is_dir():
        return work_dir if (work_dir / split).is_dir() else data_root
    if not password:
        raise FileNotFoundError(f"{archive} is password-protected: set zip_password in Config")
    tmp = work_dir / f".{split}_unpack"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    with zipfile.ZipFile(archive) as zf:
        for name in zf.namelist():
            if name.startswith(("/", "\\")) or ".." in Path(name).parts:
                raise ValueError(f"Unsafe path in {archive}: {name!r}")
        zf.extractall(tmp, pwd=password.encode())
    if not (tmp / split).is_dir():
        raise FileNotFoundError(f"{archive} has no {split}/ folder")
    os.replace(tmp / split, work_dir / split)
    shutil.rmtree(tmp, ignore_errors=True)
    n_files = sum(1 for f in (work_dir / split).rglob("*") if f.is_file())
    print(f"[unzip] {archive} -> {work_dir / split} ({n_files} file)")
    return work_dir
