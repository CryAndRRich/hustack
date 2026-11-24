import numpy as np
import pandas as pd

from pseudo import labelled

MONOTONE = {"margin": 1, "margin_prior": 1, "margin_prior_min": 1, "past_cct": 1, "p_exceed": 1, "p_exceed_prior": 1}


def lgb_fit(cfg, tr, cols, rounds=None, va=None, seed=None, monotone=False):
    import lightgbm as lgb
    params = {"objective": "binary", "learning_rate": cfg.lr, "num_leaves": cfg.leaves, "min_data_in_leaf": 40,
              "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
              "seed": cfg.seed if seed is None else seed, "verbose": -1, "num_threads": cfg.cpu_threads}
    if monotone:
        params["monotone_constraints"] = [MONOTONE.get(c, 0) for c in cols]
    dtr = lgb.Dataset(tr[cols], tr["y"], weight=tr["w"] if "w" in tr else None)
    if rounds:
        return lgb.train(params, dtr, rounds)
    dva = lgb.Dataset(va[cols], va["y"])
    return lgb.train(params, dtr, cfg.rounds, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(cfg.early_stop, verbose=False), lgb.log_evaluation(500)])


def lgb_bag(cfg, tr, cols, rounds, Xs, monotone=False):
    models = [lgb_fit(cfg, tr, cols, rounds=rounds, seed=s, monotone=monotone) for s in cfg.bag_seeds]
    return [np.mean([m.predict(X[cols]) for m in models], 0) for X in Xs]


def domain_weights(cfg, source, target, cols):
    both = pd.concat([source[cols], target[cols]], ignore_index=True).assign(
        y=np.r_[np.zeros(len(source)), np.ones(len(target))])
    p = np.clip(lgb_fit(cfg, both, cols, rounds=200).predict(source[cols]), 1e-4, 1 - 1e-4)
    w = p / (1 - p) * len(source) / len(target)
    return np.clip(w / w.mean(), 0.1, 10)


def catboost_fit_predict(cfg, tr, cols, Xs):
    from catboost import CatBoostClassifier, Pool
    model = CatBoostClassifier(iterations=cfg.cat_iters, learning_rate=cfg.cat_lr, depth=cfg.cat_depth,
                               task_type="GPU", devices="0", loss_function="Logloss", random_seed=cfg.seed,
                               verbose=0, border_count=128)
    model.fit(Pool(tr[cols], tr["y"], weight=tr["w"]))
    return [model.predict_proba(X[cols])[:, 1] for X in Xs]


def tabpfn_fit_predict(cfg, tr, cols, Xs):
    from tabpfn import TabPFNClassifier
    keep = tr["w"] >= 0.5 * tr["base_w"]
    tr = tr[keep]
    outs = [np.full(len(X), np.nan) for X in Xs]
    for cut in np.unique(np.concatenate([X["cut_s"].values for X in Xs])):
        part = tr[np.isclose(tr["cut_s"], cut)]
        if len(part) > cfg.tabpfn_max_rows:
            part = part.sample(cfg.tabpfn_max_rows, random_state=cfg.seed)
        model = TabPFNClassifier(device="cuda", n_estimators=cfg.tabpfn_estimators, ignore_pretraining_limits=True,
                                 random_state=cfg.seed)
        model.fit(part[cols].to_numpy(np.float32), part["y"].to_numpy(int))
        for X, out in zip(Xs, outs):
            m = np.isclose(X["cut_s"].values, cut)
            for start in range(0, m.sum(), cfg.tabpfn_batch):
                idx = np.where(m)[0][start:start + cfg.tabpfn_batch]
                out[idx] = model.predict_proba(X.iloc[idx][cols].to_numpy(np.float32))[:, 1]
    return outs


def protocol(cfg, fit_predict, data, log):
    rows, trunc, test, cols = data["rows"], data["trunc"], data["test"], data["cols"]
    is_val, weight = data["is_val"], data["weight"]
    out = {}
    out["val"] = fit_predict(cfg, rows[~is_val], cols, [rows[is_val]])[0]
    log("val fit done")
    oof = np.full(len(trunc), np.nan)
    for f in range(cfg.folds):
        part = trunc[trunc["fold"] != f]
        train = pd.concat([rows, labelled(part, part["q"].values, cfg, weight)], ignore_index=True)
        m = (trunc["fold"] == f).values
        oof[m] = fit_predict(cfg, train, cols, [trunc[m]])[0]
        log(f"fold {f} done")
    out["oof"] = oof
    train = pd.concat([rows, labelled(trunc, trunc["q"].values, cfg, weight)], ignore_index=True)
    out["test"] = fit_predict(cfg, train, cols, [test])[0]
    log("final fit done")
    return out
