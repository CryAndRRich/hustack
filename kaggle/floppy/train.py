import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from preprocess import load

KAGGLE = Path("/kaggle/working").exists()
OUT = Path("/kaggle/working") if KAGGLE else Path.cwd()
WORK = Path("/tmp/work") if KAGGLE else Path.cwd() / "work"
SMOKE = os.environ.get("SMOKE") == "1"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEED, BS, LR, WD, SMOOTH = 0, 256, 0.2, 5e-4, 0.1
VAL_PER_CLASS, TARGET = 50, 0.815
CONFIGS = [("r18", 1)] if SMOKE else [("r18", 30), ("r18", 60), ("wrn16_8", 30), ("r18", 100), ("wrn16_8", 60), ("wrn16_8", 100)]
DEV_BUDGET = 8 * 3600
ARCH = {"r18": (64, [(64, 1), (128, 2), (256, 2), (512, 2)]),
        "wrn16_8": (16, [(128, 1), (256, 2), (512, 2)])}
MEAN = torch.tensor([0.507, 0.487, 0.441]).view(1, 3, 1, 1) * 255
STD = torch.tensor([0.267, 0.256, 0.276]).view(1, 3, 1, 1) * 255
LOG = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def tracker_class():
    try:
        from floppy import FLOPpyTracker
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "floppy-tracker"], check=True)
        from floppy import FLOPpyTracker
    log("floppy-tracker", importlib.metadata.version("floppy-tracker"))
    return FLOPpyTracker


class Block(nn.Module):
    def __init__(self, i, o, stride):
        super().__init__()
        self.bn1, self.bn2 = nn.BatchNorm2d(i), nn.BatchNorm2d(o)
        self.c1 = nn.Conv2d(i, o, 3, stride, 1, bias=False)
        self.c2 = nn.Conv2d(o, o, 3, 1, 1, bias=False)
        self.sc = nn.Conv2d(i, o, 1, stride, bias=False) if stride != 1 or i != o else None

    def forward(self, x):
        o = F.relu(self.bn1(x))
        sc = x if self.sc is None else self.sc(o)
        return self.c2(F.relu(self.bn2(self.c1(o)))) + sc


class Net(nn.Module):
    def __init__(self, arch):
        super().__init__()
        stem, groups = ARCH[arch]
        layers, c = [nn.Conv2d(3, stem, 3, 1, 1, bias=False)], stem
        for w, s in groups:
            layers += [Block(c, w, s), Block(w, w, 1)]
            c = w
        self.body = nn.Sequential(*layers, nn.BatchNorm2d(c), nn.ReLU())
        self.fc = nn.Linear(c, 100)

    def forward(self, x):
        return self.fc(self.body(x).mean((2, 3)))


def normalize(x):
    return (x.float() - MEAN.to(x.device)) / STD.to(x.device)


def augment(x):
    b, r = len(x), torch.arange(32, device=x.device)
    xp = F.pad(x, (4, 4, 4, 4), mode="reflect")
    i, j = torch.randint(0, 9, (2, b), device=x.device)
    x = xp[torch.arange(b, device=x.device)[:, None, None, None], torch.arange(3, device=x.device)[None, :, None, None],
           (i[:, None] + r)[:, None, :, None], (j[:, None] + r)[:, None, None, :]]
    x = torch.where(torch.rand(b, 1, 1, 1, device=x.device) < 0.5, x.flip(3), x)
    cy, cx = torch.randint(0, 32, (2, b), device=x.device)
    hole = ((r[None, :, None] - cy[:, None, None]).abs() < 4) & ((r[None, None, :] - cx[:, None, None]).abs() < 4)
    return x * (~hole)[:, None]


def predict(model, x):
    model.eval()
    out = []
    with torch.no_grad(), torch.autocast(DEV, dtype=torch.float16):
        for xb in x.split(1000):
            out.append(model(normalize(xb).contiguous(memory_format=torch.channels_last)).argmax(1))
    return torch.cat(out)


def run(Tracker, arch, epochs, xa, ya, xe, ye, deadline=None):
    torch.manual_seed(SEED)
    model = Net(arch).to(DEV, memory_format=torch.channels_last)
    crit = nn.CrossEntropyLoss(label_smoothing=SMOOTH)
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, nesterov=True, weight_decay=WD)
    per_ep = len(xa) // BS
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=epochs * per_ep, pct_start=0.15)
    scaler = torch.amp.GradScaler()
    t0 = time.time()
    tracker = Tracker()
    tracker.start(model=model, optimizer=opt, criterion=crit)
    try:
        for ep in range(epochs):
            model.train()
            tot = 0.0
            for idx in torch.randperm(len(xa), device=DEV)[:per_ep * BS].view(per_ep, BS):
                xb = augment(normalize(xa[idx])).contiguous(memory_format=torch.channels_last)
                with torch.autocast(DEV, dtype=torch.float16):
                    loss = crit(model(xb), ya[idx])
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()
                tot += loss.item()
            if ep == 0 and deadline is not None and time.time() + (time.time() - t0) * epochs > deadline:
                log(f"{arch} {epochs}ep: projected past the dev budget, skipped")
                return None
            if ep % 10 == 9 or ep == epochs - 1:
                log(f"{arch} {epochs}ep | epoch {ep + 1} loss {tot / per_ep:.4f} | {time.time() - t0:.0f}s")
        pred = predict(model, xe)
        correct = (pred == ye).sum().item()
    finally:
        tracker.stop()
    rep = tracker.report()
    acc = correct / len(ye)
    log(f"{arch} {epochs}ep | acc {acc:.4f} | BOPs {rep.overall_bop:.4e} (bits/flop {rep.overall_bop / max(rep.overall_flop, 1):.2f}) "
        f"| score {acc / np.log10(rep.overall_bop):.5f} | {time.time() - t0:.0f}s")
    return acc, rep, pred.cpu().numpy()


def split(y):
    rng = np.random.default_rng(SEED)
    val = np.concatenate([rng.choice(np.flatnonzero(y == c), VAL_PER_CLASS, replace=False) for c in range(100)])
    return np.setdiff1d(np.arange(len(y)), val), val


def main():
    t_all = time.time()
    WORK.mkdir(parents=True, exist_ok=True)
    Tracker = tracker_class()
    xtr, ytr, xte, yte = load()
    if SMOKE:
        xtr, ytr = xtr[:3000], ytr[:3000]
    g = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(DEV)
    tr, va = split(ytr) if not SMOKE else (np.arange(2500), np.arange(2500, 3000))
    log("device", DEV, torch.cuda.get_device_name() if DEV == "cuda" else "", "| train", len(tr), "val", len(va))

    rows, deadline = [], time.time() + DEV_BUDGET
    for arch, ep in CONFIGS:
        r = run(Tracker, arch, ep, g(xtr[tr]), g(ytr[tr]), g(xtr[va]), g(ytr[va]), deadline)
        if r is not None:
            rows.append({"arch": arch, "epochs": ep, "val_acc": r[0], "bops": r[1].overall_bop,
                         "flops": r[1].overall_flop, "score": r[0] / np.log10(r[1].overall_bop)})
    dev = pd.DataFrame(rows)
    log("dev sweep (45k train / 5k val)\n" + dev.to_string(index=False))
    ok = dev[dev.val_acc >= TARGET]
    pick = ok.sort_values("bops").iloc[0] if len(ok) else dev.sort_values("val_acc").iloc[-1]
    log(f"pick {pick.arch} {pick.epochs}ep ({'cheapest with val_acc >= %.3f' % TARGET if len(ok) else 'no config reached the target: best val_acc'})")

    acc, rep, pred = run(Tracker, pick.arch, int(pick.epochs), g(xtr), g(ytr), g(xte), g(yte))
    bops = int(rep.overall_bop)
    log(f"FINAL test accuracy {acc * 100:.2f}% | total BOPs {bops} | score {acc / np.log10(bops):.5f}")
    log(str(rep))

    (OUT / "total_bops.txt").write_text(str(bops))
    ids = np.arange(len(pred))
    pd.DataFrame({"id": ids, "label": pred, "total_bops": bops}).to_csv(OUT / "submission.csv", index=False)
    (WORK / "extra").mkdir(exist_ok=True)
    pd.DataFrame({"id": ids, "label": pred}).to_csv(WORK / "extra" / "sub_id_label.csv", index=False)
    dev.to_csv(WORK / "dev_sweep.csv", index=False)
    (WORK / "final.json").write_text(json.dumps({"arch": pick.arch, "epochs": int(pick.epochs), "test_acc": acc,
                                                  "overall_bop": bops, "overall_flop": int(rep.overall_flop)}, indent=1))
    log(f"SUBMIT submission.csv (id,label,total_bops); fallback extra/sub_id_label.csv; total_bops.txt written | {time.time() - t_all:.0f}s")


if __name__ == "__main__":
    try:
        main()
    finally:
        WORK.mkdir(parents=True, exist_ok=True)
        (WORK / "log.txt").write_text("\n".join(LOG))
        shutil.make_archive(str(OUT / "results"), "zip", WORK)
