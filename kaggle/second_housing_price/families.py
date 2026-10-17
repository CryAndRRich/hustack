import gc
import importlib
import subprocess
import sys

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import LinearSVR

PIP = {
    "xgboost": "xgboost",
    "tabpfn": "tabpfn>=2,<3",
    "tabicl": "tabicl>=2.2,<3",
    "tabdpt": "tabdpt>=1.3,<2",
    "pytabkit": "pytabkit>=1.7,<2",
}
GPU_FAMILIES = {"tabpfn", "tabicl", "tabdpt", "realmlp", "tabm"}
KNN_COLS = ["log_pool_px", "log_pool_px_loc", "log_px_combined", "te_community", "te_street", "lat", "lng",
            "log_area", "age"]
XGB_PARAMS = dict(n_estimators=1500, learning_rate=0.02, max_depth=5, min_child_weight=3, subsample=0.8,
                  colsample_bytree=0.6, reg_lambda=1.0)
HGB_PARAMS = dict(max_iter=800, learning_rate=0.03, max_leaf_nodes=31, min_samples_leaf=15, l2_regularization=1.0)


def pip_import(module):
    try:
        return importlib.import_module(module)
    except ImportError:
        pass
    spec = PIP[module]
    print(f"  installing {spec} (needs Internet ON) ...")
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", spec], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"pip install {spec} failed: {r.stderr.strip()[-400:]}")
    importlib.invalidate_caches()
    return importlib.import_module(module)


def cuda_available():
    try:
        import torch
    except ImportError:
        return False
    return torch.cuda.is_available()


def free_gpu():
    gc.collect()
    if cuda_available():
        import torch
        torch.cuda.empty_cache()


class Design:
    def __init__(self, num_cols, cat_cols, categories):
        self.num_cols, self.cat_cols, self.categories = num_cols, cat_cols, categories

    def fit(self, X):
        num = X[self.num_cols].astype(float)
        self.median_ = num.median()
        self.flag_cols_ = [c for c in self.num_cols if num[c].isna().any()]
        Z = self._raw(X)
        self.mean_, self.std_ = Z.mean(axis=0), Z.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def _raw(self, X):
        num = X[self.num_cols].astype(float)
        parts = [num.fillna(self.median_).to_numpy(),
                 num[self.flag_cols_].isna().to_numpy(dtype=float)]
        for c in self.cat_cols:
            parts.append(np.column_stack([(X[c] == v).to_numpy(dtype=float) for v in self.categories[c]]))
        return np.column_stack(parts)

    def transform(self, X):
        return ((self._raw(X) - self.mean_) / self.std_).astype(np.float32)


def predict_family(name, Xa, ya, Xb, seed, knn_a=None, knn_b=None):
    ya = np.asarray(ya, dtype=float)
    mu, sd = ya.mean(), ya.std()
    z = (ya - mu) / sd
    back = lambda p: np.asarray(p, dtype=float).ravel() * sd + mu

    if name == "xgb_log":
        xgb = pip_import("xgboost")
        model = xgb.XGBRegressor(**XGB_PARAMS, random_state=seed, verbosity=0, n_jobs=-1)
        model.fit(Xa, np.log(ya))
        smear = np.mean(np.exp(np.log(ya) - model.predict(Xa)))
        return np.exp(model.predict(Xb)) * smear
    if name == "hgb":
        model = HistGradientBoostingRegressor(**HGB_PARAMS, random_state=seed, early_stopping=False)
        return model.fit(Xa, ya).predict(Xb)
    if name == "forest":
        pair = [RandomForestRegressor(n_estimators=400, min_samples_leaf=2, max_features=0.5, random_state=seed, n_jobs=-1),
                ExtraTreesRegressor(n_estimators=400, min_samples_leaf=2, max_features=0.7, random_state=seed, n_jobs=-1)]
        return np.mean([m.fit(Xa, ya).predict(Xb) for m in pair], axis=0)
    if name == "knn":
        model = KNeighborsRegressor(n_neighbors=15, weights="distance", n_jobs=-1)
        return model.fit(knn_a, ya).predict(knn_b)
    if name == "mlp":
        common = dict(max_iter=500, early_stopping=True, n_iter_no_change=20, validation_fraction=0.12,
                      learning_rate_init=1e-3, batch_size=64)
        nets = [MLPRegressor(hidden_layer_sizes=(256, 128), alpha=1e-2, random_state=seed, **common),
                MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-3, random_state=seed + 101, **common)]
        return back(np.mean([net.fit(Xa, z).predict(Xb) for net in nets], axis=0))
    if name == "svr":
        return back(LinearSVR(C=0.5, epsilon=0.0, max_iter=20000, dual="auto", random_state=seed).fit(Xa, z).predict(Xb))
    if name == "huber":
        return back(HuberRegressor(epsilon=1.35, alpha=1e-3, max_iter=1000).fit(Xa, z).predict(Xb))
    if name == "pls":
        n_comp = max(1, min(20, Xa.shape[1] - 1))
        return back(PLSRegression(n_components=n_comp, scale=False).fit(Xa, z).predict(Xb))
    if name == "krr":
        ny = Nystroem(kernel="rbf", gamma=1.0 / Xa.shape[1], n_components=min(1500, len(Xa)), random_state=seed)
        Za = ny.fit_transform(Xa)
        return back(Ridge(alpha=0.1).fit(Za, z).predict(ny.transform(Xb)))
    if name == "tabpfn":
        mod = pip_import("tabpfn")
        model = mod.TabPFNRegressor(device="cuda", n_estimators=8, ignore_pretraining_limits=True, random_state=seed)
        return back(model.fit(Xa, z).predict(Xb))
    if name == "tabicl":
        mod = pip_import("tabicl")
        model = mod.TabICLRegressor(random_state=seed, verbose=False, n_estimators=8)
        return back(model.fit(Xa, z).predict(Xb))
    if name == "tabdpt":
        mod = pip_import("tabdpt")
        model = mod.TabDPTRegressor(context_reduction="subsample", compile=False, use_flash=False, verbose=False)
        model.fit(Xa, z.astype(np.float32))
        return back(model.predict(Xb, n_ensembles=8, context_size=None, batch_size=1024, seed=seed))
    if name in ("realmlp", "tabm"):
        mod = pip_import("pytabkit")
        cls = mod.RealMLP_TD_Regressor if name == "realmlp" else mod.TabM_D_Regressor
        model = cls(random_state=seed, verbosity=0)
        model.fit(Xa, z.astype(np.float32))
        return back(model.predict(Xb))
    raise ValueError(name)
