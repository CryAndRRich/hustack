import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

T0 = time.time()
DEADLINE = T0 + 11.3 * 3600
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
VERIFY_MODELS = ["Qwen/Qwen3-4B", "Qwen/Qwen3-1.7B"]
LABELS = ["supports", "refutes", "NOT ENOUGH INFO"]
NEI = 2
K_CANDS = 48
N_EVID_VERIFY = 5
N_EVID_SUBMIT = 3

KAGGLE = Path("/kaggle/working").exists()
OUT = Path("/kaggle/working") if KAGGLE else Path(".")
WORK = Path("/tmp/fact_work") if KAGGLE else Path("work")
WORKERS = Path("/kaggle/working/workers") if KAGGLE else Path(__file__).resolve().parent

_SPACE = re.compile(r"\s+(?=[^\x00-\x7f])|(?<=[^\x00-\x7f])\s+")
_PAREN = re.compile(r"_?\(.*\)$")


def log(*a):
    print(f"[{(time.time() - T0) / 60:6.1f} min]", *a, flush=True)


def clean(s):
    return _SPACE.sub("", s).strip()


def match_key(s):
    return s.replace(" ", "").replace("_", "").lower()


def title_base(t):
    return match_key(_PAREN.sub("", t))


def title_display(t):
    return t.replace("_", " ")


def find_data_dir():
    roots = [Path("/kaggle/input")] if Path("/kaggle/input").exists() else []
    try:
        roots.append(Path(__file__).resolve().parent.parent / "data")
    except NameError:
        roots.append(Path.cwd() / "data")
    for root in roots:
        for dirpath, _, filenames in os.walk(root):
            if {"wiki_pages.jsonl", "test.jsonl", "train.jsonl", "dev.jsonl", "sample_submission.csv"} <= set(filenames):
                return Path(dirpath)
    raise FileNotFoundError("competition files not found: attach the competition data")


def gold_groups(r):
    if r["label"] == LABELS[NEI]:
        return []
    groups = []
    for g in r["evidence"]:
        keys = list(dict.fromkeys((e[2], int(e[3])) for e in g))
        if keys not in groups:
            groups.append(keys)
    return groups


def covered(keys, groups):
    return any(all(k in keys for k in g) for g in groups)


def tune_offsets(logp, y, ev_ok):
    def fever(b):
        pred = np.argmax(logp + b, 1)
        return np.mean((pred == y) & ((y == NEI) | ev_ok))
    best = (fever(np.zeros(3)), 0.0, 0.0)
    grid = np.round(np.arange(-3, 3.01, 0.1), 2)
    for bs in grid:
        for br in grid:
            s = fever(np.array([bs, br, 0.0]))
            if s > best[0] + 1e-9:
                best = (s, bs, br)
    return best


def load_claims(data_dir):
    rows = []
    for split in ("train", "dev", "test"):
        with open(data_dir / f"{split}.jsonl", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if split == "test":
                    assert "label" not in r and "evidence" not in r, "test rows must not carry labels"
                    rows.append({"id": int(r["id"]), "split": split, "claim": r["claim"], "y": -1, "groups": []})
                else:
                    rows.append({"id": int(r["id"]), "split": split, "claim": r["claim"],
                                 "y": LABELS.index(r["label"]), "groups": gold_groups(r)})
    df = pd.DataFrame(rows)
    counts = df["split"].value_counts().to_dict()
    assert counts == {"train": 24012, "dev": 3000, "test": 6000}, counts
    assert df["id"].is_unique
    return df


def load_wiki(path):
    titles, texts, lines = [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            titles.append(r["id"])
            texts.append(r["text"] or "")
            lines.append(r["lines"] or "")
    assert len(titles) == len(set(titles)), "duplicate page titles"
    return titles, texts, lines


def parse_lines(raw):
    out = []
    for ln in raw.split("\n"):
        parts = ln.split("\t")
        if len(parts) < 2 or not parts[0].isdigit():
            continue
        sent = clean(parts[1])
        if sent:
            out.append((int(parts[0]), sent, parts[3::2]))
    return out


def rss_gb():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20


def bigrams(s):
    return {s[i:i + 2] for i in range(len(s) - 1)}


def pip_install(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)


def run_workers(script, args, ngpu):
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={ngpu}",
           str(WORKERS / script), *map(str, args)]
    env = dict(os.environ, OMP_NUM_THREADS="2", TOKENIZERS_PARALLELISM="false",
               NCCL_P2P_DISABLE="1", NCCL_IB_DISABLE="1", PYTHONUNBUFFERED="1")
    log("launch", " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    tail = []
    for line in proc.stdout:
        print(line, end="", flush=True)
        tail = (tail + [line])[-40:]
    if proc.wait() != 0:
        raise RuntimeError(f"{script} failed with code {proc.returncode}:\n" + "".join(tail))


