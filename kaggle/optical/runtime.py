import json
import shutil
import sys
import time
from pathlib import Path

WORK = Path("/tmp/work")
_T0 = time.time()
_LOG = None


def out_dir():
    p = Path("/kaggle/working")
    return p if p.exists() else Path(".")


def data_root():
    names = ("train.csv", "validation.csv", "test.csv", "sample_submission.csv")
    for base in (Path("/kaggle/input"), Path("data/raw")):
        if base.exists():
            for p in base.rglob("validation.csv"):
                if all((p.parent / n).is_file() for n in names):
                    return p.parent
    raise FileNotFoundError("competition CSVs not found under /kaggle/input or data/raw")


def log(*parts):
    global _LOG
    if _LOG is None:
        WORK.mkdir(parents=True, exist_ok=True)
        _LOG = open(WORK / "log.txt", "a")
    line = "[%6.1fs] %s" % (time.time() - _T0, " ".join(str(p) for p in parts))
    print(line, flush=True)
    _LOG.write(line + "\n")
    _LOG.flush()


def save_json(obj, name):
    (WORK / name).write_text(json.dumps(obj, indent=1, default=float))


def save_code():
    code = WORK / "code"
    code.mkdir(parents=True, exist_ok=True)
    for p in Path(sys.path[0]).glob("*.py"):
        shutil.copy(p, code / p.name)


def zip_results():
    if _LOG is not None:
        _LOG.flush()
    shutil.make_archive(str(out_dir() / "results"), "zip", WORK)
