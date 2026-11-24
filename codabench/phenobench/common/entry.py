import importlib
import importlib.metadata
import subprocess
import sys


def start(track):
    if tuple(int(x) for x in importlib.metadata.version("transformers").split(".")[:2]) < (5, 1):
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers>=5.1,<6"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "git+https://github.com/PRBonn/phenobench.git"], check=True)
    cfg = importlib.import_module(f"{track}.config").Config()
    for k, v in vars(cfg).items():
        print(f"{k} = {v}")
    importlib.import_module(f"{track}.main").run(cfg)
