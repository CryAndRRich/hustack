import json
import os
import subprocess
import sys
import time

import numpy as np

cfg = json.load(open(sys.argv[1]))
MODEL = sys.argv[2]
PREFIX = {"tabpfn": "tab", "tabicl": "icl"}[MODEL]
subprocess.run([sys.executable, "-m", "pip", "install", "-q"] +
               (["tabpfn==2.0.9"] if MODEL == "tabpfn" else ["--no-deps", "tabicl==2.2.0", "einops"]), check=False)
import torch

torch.set_num_threads(1)
if not torch.cuda.is_available():
    sys.exit(f"{MODEL} worker: no GPU")


def make(seed):
    if MODEL == "tabpfn":
        from tabpfn import TabPFNRegressor
        return TabPFNRegressor(device="cuda", n_estimators=cfg["n_est"], random_state=seed, ignore_pretraining_limits=True)
    from tabicl import TabICLRegressor
    return TabICLRegressor(device="cuda", n_estimators=2 * cfg["n_est"], random_state=seed)


d = np.load(cfg["data"])
X, Y, folds, n_tr = d["X"], d["Y"], d["folds"], int(d["n_tr"])
Xtr, Xte = X[:n_tr], X[n_tr:]
t0 = time.time()
for r in range(len(folds)):
    for f in range(int(folds[r].max()) + 1):
        path = os.path.join(cfg["out"], f"{PREFIX}_r{r}_f{f}.npz")
        try:
            os.close(os.open(path + ".lock", os.O_CREAT | os.O_EXCL))
        except FileExistsError:
            continue
        if time.time() > cfg["deadline"]:
            sys.exit(f"{MODEL} worker: deadline before r{r} f{f}")
        tr_i, va_i = folds[r] != f, folds[r] == f
        pv, pt = np.zeros((va_i.sum(), Y.shape[1])), np.zeros((len(Xte), Y.shape[1]))
        for j in range(Y.shape[1]):
            k = tr_i & ~np.isnan(Y[:, j])
            m = make(100 * r + f)
            m.fit(Xtr[k], Y[k, j])
            pv[:, j], pt[:, j] = m.predict(Xtr[va_i]), m.predict(Xte)
        np.savez(path + ".tmp.npz", val=pv, test=pt)
        os.replace(path + ".tmp.npz", path)
        print(f"{MODEL} r{r} f{f} done {time.time() - t0:.0f}s", flush=True)
print(f"{MODEL} worker finished", flush=True)
