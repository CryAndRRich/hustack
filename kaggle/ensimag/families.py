import numpy as np
import pandas as pd

from preprocess import pip_import

CLIP_Q = 0.99
NUM_THREADS = 4
SEED = 0

LGB_PARAMS = dict(objective="binary", num_leaves=15, min_data_in_leaf=500, feature_fraction=0.7, bagging_fraction=0.7,
                  bagging_freq=1, lambda_l2=10.0, verbose=-1, seed=SEED, num_threads=NUM_THREADS)
XGB_PARAMS = dict(objective="binary:logistic", tree_method="hist", max_depth=5, min_child_weight=5.0, subsample=0.8,
                  colsample_bytree=0.7, reg_lambda=10.0, nthread=NUM_THREADS, seed=SEED, verbosity=0)
FAMILIES = ("lgbm", "xgb", "cat", "hgb", "forest", "linear", "kernel", "mlp")


def sign_weights(spread, clip_q=CLIP_Q):
    a = np.abs(spread)
    w = np.minimum(a, np.quantile(a, clip_q))
    return w / w.mean()


def standardise(xf: pd.DataFrame, xp: pd.DataFrame):

    mu, sd = xf.mean(), xf.std().replace(0.0, 1.0)
    zf = ((xf - mu) / sd).fillna(0.0).clip(-10, 10).to_numpy(np.float32)
    zp = ((xp - mu) / sd).fillna(0.0).clip(-10, 10).to_numpy(np.float32)
    return zf, zp


def fit_family(family, xf, spread_f, xp, features, cfg):
\

    y = (spread_f >= 0).astype(int)
    w = sign_weights(spread_f)
    xf, xp = xf[features], xp[features]
    if family == "lgbm":
        import lightgbm as lgb
        m = lgb.train(dict(LGB_PARAMS, learning_rate=cfg["learning_rate"]), lgb.Dataset(xf, label=y, weight=w),
                      num_boost_round=cfg["num_boost_round"])
        return m.predict(xp) - 0.5, pd.Series(m.feature_importance("gain"), index=features).sort_values(ascending=False)
    if family == "xgb":
        xgb = pip_import("xgboost")
        m = xgb.train(dict(XGB_PARAMS, eta=cfg["learning_rate"]), xgb.DMatrix(xf, label=y, weight=w),
                      num_boost_round=cfg["xgb_rounds"])
        return m.predict(xgb.DMatrix(xp)) - 0.5, None
    if family == "cat":
        cb = pip_import("catboost")
        m = cb.CatBoostClassifier(iterations=cfg["cat_iters"], learning_rate=0.05, depth=6, l2_leaf_reg=3.0,
                                  loss_function="Logloss", thread_count=NUM_THREADS, random_seed=SEED, verbose=0,
                                  allow_writing_files=False)
        m.fit(xf, y, sample_weight=w)
        return m.predict_proba(xp)[:, 1] - 0.5, None
    if family == "hgb":
        from sklearn.ensemble import HistGradientBoostingClassifier
        m = HistGradientBoostingClassifier(max_iter=cfg["hgb_iters"], learning_rate=0.05, max_leaf_nodes=31,
                                           min_samples_leaf=200, l2_regularization=1.0, early_stopping=False,
                                           random_state=SEED)
        m.fit(xf, y, sample_weight=w)
        return m.predict_proba(xp)[:, 1] - 0.5, None
    if family == "forest":
        from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
        med = xf.median()
        af, ap = xf.fillna(med).to_numpy(np.float32), xp.fillna(med).to_numpy(np.float32)
        common = dict(n_estimators=cfg["forest_trees"], min_samples_leaf=100, max_features=0.3, n_jobs=NUM_THREADS,
                      random_state=SEED)
        probs = [cls(**common).fit(af, y, sample_weight=w).predict_proba(ap)[:, 1]
                 for cls in (RandomForestClassifier, ExtraTreesClassifier)]
        return np.mean(probs, axis=0) - 0.5, None
    zf, zp = standardise(xf, xp)
    if family == "linear":
        from sklearn.linear_model import LogisticRegression
        m = LogisticRegression(C=1.0, max_iter=cfg["linear_iter"]).fit(zf, y, sample_weight=w)
        return m.predict_proba(zp)[:, 1] - 0.5, None
    if family == "kernel":
        from sklearn.kernel_approximation import Nystroem
        from sklearn.linear_model import LogisticRegression
        ny = Nystroem(kernel="rbf", gamma=1.0 / zf.shape[1], n_components=cfg["kernel_comp"], random_state=SEED)
        kf = ny.fit_transform(zf).astype(np.float32)
        m = LogisticRegression(C=1.0, max_iter=cfg["linear_iter"]).fit(kf, y, sample_weight=w)
        return m.predict_proba(ny.transform(zp).astype(np.float32))[:, 1] - 0.5, None
    if family == "mlp":
        return fit_mlp(zf, y, w, zp, cfg), None
    raise KeyError(family)


def fit_mlp(zf, y, w, zp, cfg):
    torch = pip_import("torch")
    torch.manual_seed(SEED)
    torch.set_num_threads(NUM_THREADS)
    nn = torch.nn
    net = nn.Sequential(nn.Linear(zf.shape[1], 256), nn.ReLU(), nn.Dropout(0.1),
                        nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 1))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    xt, yt, wt = torch.from_numpy(zf), torch.from_numpy(y.astype(np.float32)), torch.from_numpy(w.astype(np.float32))
    gen = torch.Generator().manual_seed(SEED)
    net.train()
    for _ in range(cfg["mlp_epochs"]):
        for idx in torch.randperm(len(xt), generator=gen).split(1024):
            opt.zero_grad()
            loss = (nn.functional.binary_cross_entropy_with_logits(net(xt[idx]).squeeze(1), yt[idx], reduction="none")
                    * wt[idx]).mean()
            loss.backward()
            opt.step()
    net.eval()
    with torch.no_grad():
        return torch.sigmoid(net(torch.from_numpy(zp)).squeeze(1)).numpy().astype(np.float64) - 0.5


def blend(preds: dict, members, weights: dict = None):

    w = np.array([1.0 / len(members) if weights is None else weights[m] for m in members])
    return np.column_stack([preds[m] / np.std(preds[m]) for m in members]) @ w


def fit_blend_weights(preds: dict, members, spread, ridge=0.05):
\

    z = np.column_stack([preds[m] / np.std(preds[m]) for m in members])
    s, sw = np.where(spread >= 0, 1.0, -1.0), sign_weights(spread)
    prior = np.full(len(members), 1.0 / len(members))
    a = prior.copy()
    lr = 1.0 / (np.linalg.norm(z * np.sqrt(sw)[:, None], 2) ** 2 / len(z) + ridge)
    for _ in range(2000):
        grad = z.T @ (sw * (z @ a - s)) / len(z) + ridge * (a - prior)
        a = project_simplex(a - lr * grad)
    return dict(zip(members, (float(v) for v in a)))


def project_simplex(v):
    u = np.sort(v)[::-1]
    css = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(v) + 1) > (css - 1))[0][-1]
    return np.maximum(v - (css[rho] - 1) / (rho + 1.0), 0.0)
