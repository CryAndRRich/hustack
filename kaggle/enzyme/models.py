import os

import numpy as np
import torch
import torch.nn as nn
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from preprocess import N_CLASSES, N_NODE_FEATS

GIN_HIDDEN = 64
GIN_LAYERS = 4
GIN_DROPOUT = 0.5
GIN_EPOCHS = 300
GIN_BATCH = 32
GIN_SEEDS = 2
GIN_DROP_EDGE = 0.1
GIN_FEAT_NOISE = 0.1
GRIDS = {"wwl": ((0.01, 0.03, 0.1, 0.3, 1.0), (100.0, 1000.0, 10000.0)),
         "fgw": ((0.1, 0.3, 1.0, 3.0), (1.0, 3.0, 10.0, 100.0))}
DISSIM_CS = (0.01, 0.1, 1.0, 10.0)
torch.set_num_threads(os.cpu_count() or 1)


def et_arm(F, y, tr, seed):
    m = ExtraTreesClassifier(n_estimators=1000, max_features="sqrt", n_jobs=-1, random_state=seed)
    return m.fit(F[tr], y[tr]).predict_proba(F)


def lgb_arm(F, y, tr, seed):
    m = LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=10, subsample=0.8,
                       subsample_freq=1, colsample_bytree=0.3, reg_lambda=1.0, random_state=seed, verbose=-1)
    return m.fit(F[tr], y[tr]).predict_proba(F)


def svm_feat_arm(F, y, tr, seed):
    m = make_pipeline(StandardScaler(), SVC(C=10, gamma="scale", probability=True, random_state=seed))
    return m.fit(F[tr], y[tr]).predict_proba(F)


def kernel_arm(K, y, tr, seed, C=10.0):
    m = SVC(kernel="precomputed", C=C, probability=True, random_state=seed)
    return m.fit(K[np.ix_(tr, tr)], y[tr]).predict_proba(K[:, tr])


def kernel_select(Ds, scales, y, folds, family):
    gammas, Cs = GRIDS[family]
    best = None
    for key, D in Ds.items():
        for g in gammas:
            K = np.exp(-g * D / scales[key])
            for C in Cs:
                pred = np.empty(len(y), dtype=np.int64)
                for tr, va in folds:
                    m = SVC(kernel="precomputed", C=C).fit(K[np.ix_(tr, tr)], y[tr])
                    pred[va] = m.predict(K[np.ix_(va, tr)])
                f1 = f1_score(y, pred, average="macro")
                if best is None or f1 > best[0]:
                    best = (f1, key, g, C)
    return best


def dissim_features(Ds, tr):
    blocks = []
    for D in Ds:
        X = D[:, tr].copy()
        own = X[tr]
        np.fill_diagonal(own, np.inf)
        np.fill_diagonal(own, own.min(1))
        X[tr] = own
        blocks.append(X)
    return np.hstack(blocks)


def dissim_fit(Ds, y, tr, C):
    X = dissim_features(Ds, tr)
    m = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))
    return m.fit(X[tr], y[tr]).predict_proba(X)


def dissim_select(Ds, y, folds):
    best = None
    for C in DISSIM_CS:
        pred = np.empty(len(y), dtype=np.int64)
        for tr, va in folds:
            pred[va] = dissim_fit(Ds, y, tr, C)[va].argmax(1)
        f1 = f1_score(y, pred, average="macro")
        if best is None or f1 > best[0]:
            best = (f1, C)
    return best


class GIN(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        dims = [d_in] + [GIN_HIDDEN] * GIN_LAYERS
        self.mlps = nn.ModuleList(
            nn.Sequential(nn.Linear(dims[i], GIN_HIDDEN), nn.BatchNorm1d(GIN_HIDDEN), nn.ReLU(),
                          nn.Linear(GIN_HIDDEN, GIN_HIDDEN), nn.BatchNorm1d(GIN_HIDDEN), nn.ReLU())
            for i in range(GIN_LAYERS))
        self.eps = nn.Parameter(torch.zeros(GIN_LAYERS))
        self.heads = nn.ModuleList(nn.Linear(2 * d, N_CLASSES) for d in dims)
        self.drop = nn.Dropout(GIN_DROPOUT)

    def forward(self, x, src, dst, batch, n_graphs):
        cnt = torch.bincount(batch, minlength=n_graphs).clamp(min=1).unsqueeze(1).float()
        h, out = x, 0
        for i, head in enumerate(self.heads):
            if i:
                agg = torch.zeros_like(h).index_add_(0, dst, h[src])
                h = self.mlps[i - 1]((1 + self.eps[i - 1]) * h + agg)
            s = torch.zeros(n_graphs, h.shape[1]).index_add_(0, batch, h)
            out = out + head(self.drop(torch.cat([s, s / cnt], 1)))
        return out


def collate(graphs, idx):
    xs, src, dst, batch, off = [], [], [], [], 0
    for k, i in enumerate(idx):
        g = graphs[i]
        n = len(g["hg"])
        xs.append(g["hg"])
        src.append(g["e"][0] + off)
        dst.append(g["e"][1] + off)
        batch.append(np.full(n, k))
        off += n
    t = lambda a, dt: torch.as_tensor(np.concatenate(a), dtype=dt)
    return t(xs, torch.float32), t(src, torch.long), t(dst, torch.long), t(batch, torch.long), len(idx)


def gin_one(graphs, y, tr, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = GIN(graphs[0]["hg"].shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, GIN_EPOCHS)
    yt = torch.as_tensor(y, dtype=torch.long)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(GIN_EPOCHS):
        model.train()
        perm = rng.permutation(tr)
        for s in range(0, len(perm), GIN_BATCH):
            chunk = perm[s:s + GIN_BATCH]
            if len(chunk) < 2:
                continue
            x, src, dst, batch, ng = collate(graphs, chunk)
            keep = torch.rand(src.shape[0]) >= GIN_DROP_EDGE
            x[:, :N_NODE_FEATS] += GIN_FEAT_NOISE * torch.randn(x.shape[0], N_NODE_FEATS)
            opt.zero_grad()
            loss_fn(model(x, src[keep], dst[keep], batch, ng), yt[chunk]).backward()
            opt.step()
        sched.step()
    model.eval()
    with torch.no_grad():
        return np.vstack([torch.softmax(model(*collate(graphs, np.arange(s, min(s + 64, len(graphs))))), 1).numpy()
                          for s in range(0, len(graphs), 64)])


def gin_arm(graphs, y, tr, seed):
    return np.mean([gin_one(graphs, y, tr, seed * 100 + k) for k in range(GIN_SEEDS)], axis=0)
