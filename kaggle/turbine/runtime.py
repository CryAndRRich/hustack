import csv
import os
import resource
import shutil
import sys
import time
import zipfile
from collections import Counter


import config

INPUT_ROOTS = ["/kaggle/input", "../input", "./input", "."]
_T0 = time.time()
_LOG = None


def out_dir():
    return "/kaggle/working" if os.path.isdir("/kaggle/working") else "."


def init():
    global _LOG
    shutil.rmtree(config.WORK, ignore_errors=True)
    os.makedirs(os.path.join(config.WORK, "extra"), exist_ok=True)
    _LOG = open(os.path.join(config.WORK, "log.txt"), "w")


def log(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True)
    if _LOG:
        _LOG.write(line + "\n")
        _LOG.flush()


def mem():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_gb = peak / 1024 ** 2 if sys.platform.startswith("linux") else peak / 1024 ** 3
    cur = avail = float("nan")
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    cur = int(line.split()[1]) / 1024 ** 2
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    avail = int(line.split()[1]) / 1024 ** 2
    except OSError:
        pass
    return "rss=%.1fGB peak=%.1fGB avail=%.1fGB" % (cur, peak_gb, avail)


def step(msg):
    log("  [%6.0fs %s] %s" % (time.time() - _T0, mem(), msg))


def section(title):
    log("\n" + "=" * 78 + "\n" + title + "\n" + "=" * 78)
    step("start")


def find_inputs():
    for root in INPUT_ROOTS:
        if os.path.isdir(root) and [e for e in os.listdir(root) if not e.startswith(".")]:
            found = {}
            for dirpath, _d, names in os.walk(root):
                for fn in names:
                    if fn.lower().endswith(".csv"):
                        found[fn.lower()] = os.path.join(dirpath, fn)
            files = {"train": _pick(found, "training", exclude=("catalog",)),
                     "cand": _pick(found, "candidates", exclude=("catalog",)),
                     "cat_tr": _pick(found, "catalog", "training"),
                     "cat_ca": _pick(found, "catalog", "candidates"),
                     "mapping": _pick(found, "mapping")}
            missing = [k for k, v in files.items() if v is None]
            if missing:
                raise FileNotFoundError("inputs not found: %s" % missing)
            return files
    raise FileNotFoundError("no input directory among %s" % INPUT_ROOTS)


def _pick(found, *needles, exclude=()):
    hits = sorted((len(n), p) for n, p in found.items()
                  if all(x in n for x in needles) and not any(x in n for x in exclude))
    return hits[0][1] if hits else None


def dialect(path, rows=50_000):
    def aligned(kwargs):
        csv.field_size_limit(min(sys.maxsize, 2 ** 31 - 1))
        widths = Counter()
        with open(path, newline="", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f, **kwargs)
            n = len(next(reader))
            for i, row in enumerate(reader):
                if i >= rows:
                    break
                widths[len(row)] += 1
        return widths.get(n, 0) / max(sum(widths.values()), 1)

    plain, esc = aligned({}), aligned({"escapechar": "\\"})
    log("  dialect %-44s default ok=%.4f escapechar ok=%.4f" % (os.path.basename(path), plain, esc))
    if plain == 1.0:
        return {}
    if esc == 1.0:
        return {"escapechar": "\\"}
    raise ValueError("no dialect aligns every row of %s" % path)


def finish(code_dir):
    if _LOG:
        _LOG.flush()
    path = os.path.join(out_dir(), "results.zip")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for dirpath, _d, names in os.walk(config.WORK):
            for fn in names:
                full = os.path.join(dirpath, fn)
                z.write(full, os.path.relpath(full, config.WORK))
        if code_dir and os.path.isdir(code_dir):
            for fn in sorted(os.listdir(code_dir)):
                if fn.endswith(".py"):
                    z.write(os.path.join(code_dir, fn), os.path.join("code", fn))
    print("results: %s" % path, flush=True)
