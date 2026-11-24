import importlib.util
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
T0 = time.time()


def log(*a, tag=""):
    print(f"[{tag}{time.time() - T0:7.1f}s]", *a, flush=True)


def walk(suffix):
    for root, _, files in os.walk(INPUT):
        for f in files:
            if f.endswith(suffix):
                yield Path(root) / f


def find(name):
    path = next((p for p in walk(name) if p.name == name), None)
    assert path is not None, f"{name} not found under {INPUT}"
    return path


def as_text(s):
    return s.astype(str).fillna("nan")


def pip_install(module, package):
    if importlib.util.find_spec(module) is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", package], check=True)


def stream(proc, prefix):
    for line in proc.stdout:
        print(f"{prefix} {line}", end="", flush=True)


def run_workers(jobs):
    started = []
    for cmd, gpu, prefix in jobs:
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONUNBUFFERED": "1"}
        p = subprocess.Popen([str(c) for c in cmd], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, bufsize=1)
        t = threading.Thread(target=stream, args=(p, prefix))
        t.start()
        started.append((prefix, p, t))
    for prefix, p, t in started:
        t.join()
        assert p.wait() == 0, f"{prefix} worker failed (output above)"
