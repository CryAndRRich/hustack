import os

import lightgbm as lgb
import numpy as np
import torch
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer, SplineTransformer
from torch import nn

YSCALE = 30.0

class Net(nn.Module):
    def __init__(self, n_num, levels):
        super().__init__()
        self.emb = nn.ModuleList([nn.Embedding(n + 1, min(16, (n + 1) // 2 + 1)) for n in levels])
        width = n_num + sum(e.embedding_dim for e in self.emb)
        self.mlp = nn.Sequential(nn.Linear(width, 256), nn.SiLU(), nn.Dropout(0.2), nn.Linear(256, 128), nn.SiLU(),
                                 nn.Dropout(0.2), nn.Linear(128, 1))

    def forward(self, xn, xc):
        return self.mlp(torch.cat([xn] + [e(xc[:, i]) for i, e in enumerate(self.emb)], 1)).squeeze(1)

class TabMLP:
    def __init__(self, cats, seed, epochs, batch=1024, lr=1e-3, patience=8):
        self.cats, self.seed, self.epochs, self.batch, self.lr, self.patience = cats, seed, epochs, batch, lr, patience

    def encode(self, X, fit=False):
        num = X.drop(columns=self.cats).to_numpy(np.float32)
        if fit:
            self.nan_cols = np.isnan(num).any(0)
            self.qt = QuantileTransformer(n_quantiles=200, output_distribution="normal", random_state=self.seed).fit(num)
        z = self.qt.transform(num)
        xn = np.hstack([np.nan_to_num(z), np.isnan(z[:, self.nan_cols])]).astype(np.float32)
        xc = np.stack([X[c].cat.codes.to_numpy() + 1 for c in self.cats], 1).astype(np.int64)
        return torch.from_numpy(xn), torch.from_numpy(xc)

    def fit(self, X, y, w, Xm, ym):
        torch.manual_seed(self.seed)
        torch.set_num_threads(os.cpu_count() or 1)
        xn, xc = self.encode(X, fit=True)
        mn, mc = self.encode(Xm)
        yt, wt = torch.tensor(y / YSCALE, dtype=torch.float32), torch.tensor(w, dtype=torch.float32)
        self.net = Net(xn.shape[1], [len(X[c].cat.categories) for c in self.cats])
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
        best, best_state, wait, g = np.inf, None, 0, torch.Generator().manual_seed(self.seed)
        for epoch in range(self.epochs):
            self.net.train()
            for idx in torch.randperm(len(yt), generator=g).split(self.batch):
                loss = (wt[idx] * (self.net(xn[idx], xc[idx]) - yt[idx]) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            val = float(np.mean((self.raw(mn, mc) - ym) ** 2))
            sched.step(val)
            if val < best:
                best, wait, self.best_epoch = val, 0, epoch
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
            else:
                wait += 1
                if wait >= self.patience:
                    break
        self.net.load_state_dict(best_state)
        return self

    def raw(self, xn, xc):
        self.net.eval()
        with torch.no_grad():
            return np.concatenate([self.net(a, b).numpy() for a, b in zip(xn.split(8192), xc.split(8192))]) * YSCALE

    def predict(self, X):
        return self.raw(*self.encode(X))

class Forest:
    def __init__(self, kind, cats, seed, n):
        self.kind, self.cats, self.seed, self.n = kind, cats, seed, n

    def mat(self, X):
        Z = X.copy()
        for c in self.cats:
            Z[c] = Z[c].cat.codes
        return np.nan_to_num(Z.to_numpy(np.float32), nan=-999.0)

    def fit(self, X, y, w, Xm, ym):
        est = RandomForestRegressor if self.kind == "rf" else ExtraTreesRegressor
        self.m = est(n_estimators=self.n, max_features=0.33 if self.kind == "rf" else 0.5,
                     min_samples_leaf=5 if self.kind == "rf" else 3, n_jobs=-1, random_state=self.seed)
        self.m.fit(self.mat(X), y, sample_weight=w)
        return self

    def predict(self, X):
        return self.m.predict(self.mat(X))

class SplineRidge:
    def __init__(self, cats, seed):
        self.cats, self.seed = cats, seed

    def design(self, X, fit=False):
        num = X.drop(columns=self.cats).to_numpy(float)
        if fit:
            self.keep = np.nanmax(np.where(np.isnan(num), -np.inf, num), 0) > np.nanmin(np.where(np.isnan(num), np.inf, num), 0)
        num = num[:, self.keep]
        if fit:
            self.nan_cols = np.isnan(num).any(0)
            self.qt = QuantileTransformer(n_quantiles=200, random_state=self.seed).fit(num)
        z = self.qt.transform(num)
        flags = np.isnan(z[:, self.nan_cols])
        z = np.nan_to_num(z, nan=0.5)
        if fit:
            self.sp = SplineTransformer(n_knots=6, degree=3).fit(z)
            self.oh = OneHotEncoder(handle_unknown="ignore", min_frequency=20, sparse_output=False).fit(X[self.cats].astype(str))
        return np.hstack([self.sp.transform(z), flags, self.oh.transform(X[self.cats].astype(str))])

    def fit(self, X, y, w, Xm, ym):
        self.m = RidgeCV(alphas=np.logspace(-1, 3, 9)).fit(self.design(X, fit=True), y, sample_weight=w)
        return self

    def predict(self, X):
        return self.m.predict(self.design(X))

class Hurdle:
    def __init__(self, params, seed, n):
        self.params, self.seed, self.n = params, seed, n

    def fit(self, X, y, w, Xm, ym):
        stop = [lgb.early_stopping(200, verbose=False)]
        self.c = lgb.LGBMClassifier(**self.params, metric="binary_logloss", n_estimators=self.n, random_state=self.seed)
        self.c.fit(X, y > 0, sample_weight=w, eval_set=[(Xm, ym > 0)], callbacks=stop)
        pos, mpos = y > 0, ym > 0
        self.r = lgb.LGBMRegressor(**self.params, metric="l2", n_estimators=self.n, random_state=self.seed)
        self.r.fit(X[pos], y[pos], sample_weight=w[pos], eval_set=[(Xm[mpos], ym[mpos])], callbacks=stop)
        self.best = self.c.best_iteration_
        return self

    def predict(self, X):
        return self.c.predict_proba(X)[:, 1] * self.r.predict(X)
