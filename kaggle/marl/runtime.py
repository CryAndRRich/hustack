import json
import os
import shutil
import sys
import time
from pathlib import Path

WORK = Path("/tmp/work")
_T0 = time.time()
_LOG = None


def elapsed():
    return time.time() - _T0


def out_dir():
    p = Path("/kaggle/working")
    return p if p.exists() else Path(".")


def data_root():
    for base in (Path("/kaggle/input"), Path("data/raw")):
        if base.exists():
            for p in base.rglob("engine.py"):
                root = p.parent.parent
                if p.parent.name == "marl1d" and all((root / n).is_file() for n in
                                                     ("train.csv", "test.csv", "sample_submission.csv")):
                    return root
    raise FileNotFoundError("train.csv + marl1d/engine.py not found under /kaggle/input or data/raw")


def import_engine(root):
    dst = Path("/tmp/engine")
    if not (dst / "marl1d").exists():
        shutil.copytree(root / "marl1d", dst / "marl1d", ignore=shutil.ignore_patterns("tests", "__pycache__"))
    if str(dst) not in sys.path:
        sys.path.append(str(dst))


def log(*parts):
    global _LOG
    if _LOG is None:
        WORK.mkdir(parents=True, exist_ok=True)
        _LOG = open(WORK / "log.txt", "a")
    line = "[%7.1fs] %s" % (elapsed(), " ".join(str(p) for p in parts))
    print(line, flush=True)
    _LOG.write(line + "\n")
    _LOG.flush()


def save_json(obj, name):
    (WORK / name).write_text(json.dumps(obj, indent=1, default=float))


def save_code():
    code = WORK / "code"
    code.mkdir(parents=True, exist_ok=True)
    for p in Path(__file__).resolve().parent.glob("*.py"):
        shutil.copy(p, code / p.name)


def zip_results():
    if _LOG is not None:
        _LOG.flush()
    shutil.make_archive(str(out_dir() / "results"), "zip", WORK)


def n_cpu():
    return max(1, len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1)
