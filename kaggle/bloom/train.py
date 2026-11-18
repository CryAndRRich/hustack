import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from preprocess import features, load

OUT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
SMOKE = os.environ.get("SMOKE") == "1"
K, SEED = 5, 0
CNN_SEEDS, CNN_EPOCHS = (1, 2) if SMOKE else (3, 60)
BUDGET = {"logreg": 300, "lgb": 900, "tabpfn": 2400, "cnn": 2400}
DEV = "cuda" if torch.cuda.is_available() else "cpu"


class Timeout(Exception):
    pass


def alarm(_sig, _frm):
    raise Timeout()


signal.signal(signal.SIGALRM, alarm)


def scores(y, p):
    return accuracy_score(y, p.argmax(1)), f1_score(y, p.argmax(1), average="macro")


def prior_cut_probe(F, y, Fv, yv):
    cum = np.cumsum(np.bincount(y, minlength=4))[:-1] / len(y)
    rows = []
    for c in F.columns:
        best = None
        for s in (1, -1):
            cuts = np.quantile(s * F[c], cum)
            acc = accuracy_score(y, np.searchsorted(cuts, s * F[c]))
            if best is None or acc > best[0]:
                best = (acc, accuracy_score(yv, np.searchsorted(cuts, s * Fv[c])))
        rows.append((c, *best))
    t = pd.DataFrame(rows, columns=["feature", "train_acc", "val_acc"]).sort_values("train_acc", ascending=False)
    print("exact-structure probe: one feature cut at the class priors (ordinal)\n", t.head(10).to_string(index=False))


def fit_logreg(Xa, ya, Xs, _seed):
    m = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=3000)).fit(Xa, ya)
    return [m.predict_proba(x) for x in Xs]


def fit_lgb(Xa, ya, Xs, seed):
    m = lgb.LGBMClassifier(objective="multiclass", n_estimators=50 if SMOKE else 400, learning_rate=0.03,
                           num_leaves=15, min_child_samples=10, colsample_bytree=0.5, subsample=0.8,
                           subsample_freq=1, reg_lambda=1.0, random_state=seed, verbose=-1).fit(Xa, ya)
    return [m.predict_proba(x) for x in Xs]


def tabpfn_classifier():
    if "tabpfn" not in sys.modules:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tabpfn==2.0.9"], check=False)
    from tabpfn import TabPFNClassifier
    return TabPFNClassifier


def fit_tabpfn(Xa, ya, Xs, seed):
    m = tabpfn_classifier()(device=DEV, n_estimators=2 if SMOKE else 8, random_state=seed,
                            ignore_pretraining_limits=True).fit(Xa, ya)
    return [m.predict_proba(x) for x in Xs]


class Net(nn.Module):
    def __init__(self, b=11, w=48):
        super().__init__()
        blk = lambda i, o: nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.GELU())
        self.conv = nn.Sequential(blk(b, w), blk(w, w), nn.MaxPool2d(2), blk(w, 2 * w), blk(2 * w, 2 * w))
        self.spec = nn.Sequential(nn.Linear(b, w), nn.GELU())
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(3 * w, 4))

    def forward(self, x):
        return self.head(torch.cat([self.conv(x).mean((2, 3)), self.spec(x.mean((2, 3)))], 1))


def d4(x, k):
    x = torch.rot90(x, k % 4, (2, 3))
    return torch.flip(x, (3,)) if k >= 4 else x


def fit_cnn(Xa, ya, Xs, seed):
    torch.manual_seed(seed)
    mu, sd = Xa.mean((0, 1, 2)), Xa.std((0, 1, 2)) + 1e-6
    t = lambda x: torch.tensor(((x - mu) / sd).transpose(0, 3, 1, 2), dtype=torch.float32, device=DEV)
    xa, yt = t(Xa), torch.tensor(ya, device=DEV)
    net = Net().to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-2)
    steps = CNN_EPOCHS * int(np.ceil(len(xa) / 64))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.05)
    g = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(CNN_EPOCHS):
        net.train()
        for idx in torch.randperm(len(xa), generator=g).split(64):
            xb = d4(xa[idx.to(DEV)], int(torch.randint(8, (1,), generator=g)))
            loss = lossf(net(xb), yt[idx.to(DEV)])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
    net.eval()
    with torch.no_grad():
        return [np.mean([torch.softmax(net(d4(t(x), k)), 1).cpu().numpy() for k in range(8)], 0) for x in Xs]


def run_arm(name, fit, Xtr, y, others, seeds):
    t0 = time.time()
    signal.alarm(BUDGET[name])
    oof = np.zeros((len(y), 4))
    rest = [np.zeros((len(o), 4)) for o in others]
    folds = list(StratifiedKFold(K, shuffle=True, random_state=SEED).split(Xtr, y))
    try:
        for f, (a, v) in enumerate(folds):
            for s in seeds:
                sub = Xtr.iloc if isinstance(Xtr, pd.DataFrame) else Xtr
                out = fit(sub[a], y[a], [sub[v]] + others, s)
                oof[v] += out[0] / len(seeds)
                for r, o in zip(rest, out[1:]):
                    r += o / len(seeds) / K
            if f == 0 and (time.time() - t0) * K > BUDGET[name]:
                print(f"{name}: projected {(time.time() - t0) * K:.0f}s > budget, skipped")
                return None
    except Exception as e:
        print(f"{name}: failed after {time.time() - t0:.0f}s: {type(e).__name__}: {e}")
        return None
    finally:
        signal.alarm(0)
    print(f"{name}: {time.time() - t0:.0f}s | OOF acc {scores(y, oof)[0]:.4f} macroF1 {scores(y, oof)[1]:.4f}")
    return oof, rest


def stack_weights(P, y):
    n = len(P)
    obj = lambda w: -np.log(np.tensordot(w, P, 1)[np.arange(len(y)), y].clip(1e-9)).mean()
    r = minimize(obj, np.full(n, 1 / n), method="SLSQP", bounds=[(0, 1)] * n,
                 constraints={"type": "eq", "fun": lambda w: w.sum() - 1})
    return r.x


def f1_class_weights(p, y):
    w = np.ones(4)
    for _ in range(3):
        for c in range(1, 4):
            grid = np.exp(np.linspace(-1.5, 1.5, 31))
            best = max(grid, key=lambda g: f1_score(y, (p * np.r_[w[:c], g, w[c + 1:]]).argmax(1), average="macro"))
            w[c] = best
    return w


if __name__ == "__main__":
    t_all = time.time()
    X, Y, ids = load()
    if SMOKE:
        X = {p: x[:200] if p == "train" else x for p, x in X.items()}
        Y["train"] = Y["train"][:200]
    y, yv = Y["train"], Y["val"]
    F = {p: features(x) for p, x in X.items()}
    print("features", F["train"].shape, "device", DEV)
    prior_cut_probe(F["train"], y, F["val"], yv)

    tab_parts = [F["val"], F["eval"]]
    img_parts = [X["val"], X["eval"]]
    arms = {"logreg": (fit_logreg, F["train"], tab_parts, [SEED]),
            "lgb": (fit_lgb, F["train"], tab_parts, [SEED, SEED + 1]),
            "tabpfn": (fit_tabpfn, F["train"], tab_parts, [SEED]),
            "cnn": (fit_cnn, X["train"], img_parts, list(range(CNN_SEEDS)))}
    res = {}
    for name, (fit, Xtr, others, seeds) in arms.items():
        r = run_arm(name, fit, Xtr, y, others, seeds)
        if r is not None:
            res[name] = r
    assert res, "every arm failed"

    names = list(res)
    w = stack_weights(np.stack([res[n][0] for n in names]), y)
    blend = lambda i: sum(wi * (res[n][0] if i is None else res[n][1][i]) for wi, n in zip(w, names))
    oof, pv, pe = blend(None), blend(0), blend(1)
    cw = f1_class_weights(oof, y)
    table = []
    for n in names + ["stack"]:
        o, v = (oof, pv) if n == "stack" else (res[n][0], res[n][1][0])
        table.append([n, *scores(y, o), *scores(yv, v), f1_score(y, (o * cw).argmax(1), average="macro")])
    print("stack weights", dict(zip(names, np.round(w, 3))), "| macro-F1 class weights", np.round(cw, 2))
    print(pd.DataFrame(table, columns=["arm", "oof_acc", "oof_f1", "val_acc", "val_f1", "oof_f1_cw"]).round(4).to_string(index=False))

    (OUT / "extra").mkdir(exist_ok=True)
    sub = lambda p: pd.DataFrame({"id": ids, "target": p.astype(int)})
    sub(pe.argmax(1)).to_csv(OUT / "submission.csv", index=False)
    sub((pe * cw).argmax(1)).to_csv(OUT / "extra" / "stack_macro_f1.csv", index=False)
    sub(np.ones(len(ids))).to_csv(OUT / "extra" / "probe_all_ones.csv", index=False)
    for n in names:
        sub(res[n][1][1].argmax(1)).to_csv(OUT / "extra" / f"{n}.csv", index=False)
    np.savez(OUT / "preds.npz", names=names, weights=w, cw=cw,
             **{f"{n}_{k}": a for n in names for k, a in zip(["oof", "val", "eval"], [res[n][0], *res[n][1]])})
    print("eval class counts: argmax", np.bincount(pe.argmax(1), minlength=4), "| macro-F1", np.bincount((pe * cw).argmax(1), minlength=4))
    print(f"SUBMIT submission.csv = stack argmax (Accuracy). If the all-ones probe shows Macro F1, submit extra/stack_macro_f1.csv | {time.time() - t_all:.0f}s")
