import json
import shutil
import sys
import time
import zipfile
from dataclasses import asdict
from pathlib import Path


class Tee:
    def __init__(self, path):
        self.file = open(path, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, text):
        self.stdout.write(text)
        self.file.write(text)
        self.file.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()


def find_folder(cfg, anchor, accept):
    for root in [Path("/kaggle/input"), Path(cfg.local_data)]:
        if not root.exists():
            continue
        for path in sorted(root.rglob(anchor)):
            if accept(path.parent):
                return path.parent
    raise FileNotFoundError(f"no input folder with {anchor} for this track")


def zip_results(cfg, work, out_dir):
    target = out_dir / "results.zip"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(work.rglob("*")):
            if p.is_file():
                z.write(p, p.relative_to(work))
        code = Path(cfg.code_dir)
        for p in sorted(code.rglob("*.py")) if code.exists() else []:
            z.write(p, Path("code") / p.relative_to(code))
    print(f"wrote {target}")


def run_session(cfg, solve):
    out_dir = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
    work = Path(cfg.work_dir)
    shutil.rmtree(work, ignore_errors=True)
    (work / "extra").mkdir(parents=True)
    old_stdout = sys.stdout
    sys.stdout = Tee(work / "log.txt")
    start = time.time()
    submission = out_dir / cfg.submission_name
    try:
        print("config", json.dumps(asdict(cfg), indent=1))
        (work / "config.json").write_text(json.dumps(asdict(cfg), indent=1))
        result = solve(cfg, work, submission)
        result["seconds"] = round(time.time() - start, 1)
        (work / "result.json").write_text(json.dumps(result, indent=1))
        print(f"SUBMIT: {submission}")
    finally:
        sys.stdout.file.close()
        sys.stdout = old_stdout
        zip_results(cfg, work, out_dir)
