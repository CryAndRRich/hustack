import json
import os
import sys
import time
import zipfile
from pathlib import Path

T0 = time.time()


class Tee:
    def __init__(self, path):
        self.file = open(path, "a", buffering=1, encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, text):
        self.stdout.write(text)
        self.file.write(text)

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def start_log(work):
    Path(work).mkdir(parents=True, exist_ok=True)
    sys.stdout = Tee(Path(work) / "log.txt")


def log(*parts):
    print(f"[{(time.time() - T0) / 60:7.1f} min]", *parts, flush=True)


def output_dir():
    return Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")


def find_files(name_test, root="/kaggle/input"):
    found = []
    for d, _, files in os.walk(root):
        found += [Path(d) / f for f in files if name_test(f)]
    return sorted(found)


def write_results(work, code_dir, extra_files, summary):
    work = Path(work)
    (work / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    members = [(p.name, p) for p in (work / "log.txt", work / "summary.json") if p.exists()]
    members += [(arc, Path(src)) for arc, src in extra_files if Path(src).exists()]
    members += [(f"code/{p.name}", p) for p in sorted(Path(code_dir).glob("*.py"))]
    sys.stdout.flush()
    with zipfile.ZipFile(output_dir() / "results.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for arc, src in members:
            z.write(src, arc)
