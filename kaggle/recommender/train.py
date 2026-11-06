import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from content import external, item_side, user_week
from models import ERA_GENERATORS, GENERATORS, STATIC, USER_GENERATORS, cosine, finetune, pretrain
from preprocess import LIKE, load, matrix, split
from rank import candidates, cross_validate, features, fit, labels, metrics, recommend, top_from_pairs

VERSION = "V5"
SELECT = "ndcg"
OUT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
T_START = time.time()
LAUNCH_MARGIN = 52
FULL = {
    "grids": {"ease": [{"lam": 800}],
              "rp3": [{"alpha": 0.6, "beta": 0.35, "k": 300}],
              "content": [{"k": 50}],
              "ease_ext": [{"lam": v} for v in (8000, 16000, 32000, 64000)],
              "ease_like_ext": [{"lam": v} for v in (8000, 16000, 32000, 64000)],
              "edlae_ext": [{"lam": 100, "drop": d} for d in (0.5, 0.67, 0.8)],
              "trans": [{"k": 500, "beta": 0.0, "steps": st, "session": se} for st in (5, 20) for se in (0, 3600)],
              "timepop": [{"weeks": w} for w in (1, 2)],
              "ease_like_era": [{"lam": v, "tau": 10} for v in (8000, 32000)],
              "trans_era": [{"k": 500, "steps": 20, "session": 3600, "tau": t} for t in (2, 3)]},
    "taus": (1, 2, 3, 10, 30), "cuts": (0.5, 0.6, 0.7, 0.8), "sas_pre_epochs": 5, "sas_pre_users": None,
    "sas_ft": (0, 5, 20), "rounds": 3000, "rank_seeds": (0, 1, 2),
}
SMOKE = {"grids": {k: v[:1] for k, v in FULL["grids"].items()}, "taus": (10,), "cuts": (0.7, 0.8), "sas_pre_epochs": 1,
         "sas_pre_users": 3000, "sas_ft": (0, 1), "rounds": 20, "rank_seeds": (0, 1)}


def log(name, cfg, m):
    print(f"  {name:9s} {cfg} " + " ".join(f"{k} {v:.4f}" for k, v in m.items()) + f" ({time.time() - T_START:.0f}s)",
          flush=True)


def context(df, n_users, E, ext):
    week = user_week(df, n_users, ext["t0"])
    return {"X": matrix(df, n_users), "Y": matrix(df[df.rating >= LIKE], n_users), "E": E, "ext": ext, "week": week,
            "df": df, "n_users": n_users, "allowed": ext["launch"][None, :] <= week[:, None] + LAUNCH_MARGIN}


def item_item(ctx, name, params):
    if name not in STATIC:
        return GENERATORS[name](ctx, **params)
    key = (name, tuple(sorted(params.items())))
    cache = ctx["ext"]["cache"]
    if key not in cache:
        cache[key] = GENERATORS[name](ctx, **params)
    return cache[key]


def tune(df, n_users, E, ext, targets, cfg):
    ctx = context(df, n_users, E, ext)
    Xt = {tau: matrix(df, n_users, tau) for tau in cfg["taus"]}
    best = {}
    for name, grid in cfg["grids"].items():
        for params in grid:
            if name in USER_GENERATORS:
                runs = [(None, USER_GENERATORS[name](ctx, **params))]
            else:
                W = GENERATORS[name](ctx, **params)
                runs = [(tau, Xt[tau] @ W) for tau in cfg["taus"]]
            for tau, s in runs:
                m = metrics(recommend(s, ctx["X"], ctx["allowed"]), targets)
                log(name, params if tau is None else {**params, "tau": tau}, m)
                if name not in best or m[SELECT] > best[name][2][SELECT]:
                    best[name] = (params, tau, m, s)
    cache = ext["cache"]
    for key in [k for k in cache if k[0] in ERA_GENERATORS]:
        params = dict(best[key[0]][0])
        if dict(key[1]) != {p: v for p, v in params.items() if p != "tau"}:
            del cache[key]
    return best, ctx


def generate(df, n_users, E, ext, best, state, ft):
    ctx = context(df, n_users, E, ext)
    scores = {}
    for name, (params, tau, _, _) in best.items():
        if name in USER_GENERATORS:
            scores[name] = USER_GENERATORS[name](ctx, **params)
        else:
            scores[name] = matrix(df, n_users, tau) @ item_item(ctx, name, params)
    scores["sasrec"] = finetune(state, df, n_users, E, (ft,))[ft]
    return scores, ctx


def write(te, top, extra):
    def frame(t):
        sub = pd.DataFrame({"id": te.id, "recommendations": [" ".join(map(str, t[u])) for u in te.userId]})
        assert len(sub) == 588 and all(len(set(r.split())) == 20 for r in sub.recommendations)
        return sub

    path = OUT / "submission.csv"
    frame(top).to_csv(path, index=False)
    d = OUT / "extra" / VERSION
    d.mkdir(parents=True, exist_ok=True)
    for name, t in extra.items():
        frame(t).to_csv(d / f"{name}.csv", index=False)
    return path


def run(cfg, tr, te, n_users, year, E, ext):
    ext["cache"].clear()
    main = cfg["cuts"][-1]
    early, targets = split(tr, main)
    best, ctx = tune(early, n_users, E, ext, targets, cfg)
    state = pretrain(ext, E, cfg["sas_pre_epochs"], cfg["sas_pre_users"])
    ft_scores = finetune(state, early, n_users, E, cfg["sas_ft"])
    ft_ndcg = {c: metrics(recommend(s, ctx["X"], ctx["allowed"]), targets)[SELECT] for c, s in ft_scores.items()}
    print("  sasrec fine-tune epochs -> ndcg: " + ", ".join(f"{c}: {v:.4f}" for c, v in ft_ndcg.items()), flush=True)
    ft = max(ft_ndcg, key=ft_ndcg.get)

    parts, single = [], {}
    for k, q in enumerate(cfg["cuts"]):
        early, targets = split(tr, q)
        scores, ctx = generate(early, n_users, E, ext, best, state, ft)
        X, allowed = ctx["X"], ctx["allowed"]
        if q == main:
            single = {name: metrics(recommend(s, X, allowed), targets) for name, s in scores.items()}
            for name, m in single.items():
                log(name, best[name][:2] if name in best else {"pretrain": cfg["sas_pre_epochs"], "finetune": ft}, m)
        users = np.array(sorted(targets))
        u, i = candidates(scores, X, allowed, users)
        y = labels(u, i, targets)
        print(f"cut {q} candidates: {len(u)} pairs, {len(u) / len(users):.0f} per user, "
              f"recall {y.sum() / sum(len(t) for t in targets.values()):.4f} ({time.time() - T_START:.0f}s)", flush=True)
        f = features(early, X, scores, {"S": cosine(X), "E": E, "year": year, "ext": ext, "week": ctx["week"],
                                        "allowed": allowed}, u, i)
        parts.append((f, y, u, i, np.full(len(u), k)))
    F = pd.concat([p[0] for p in parts], ignore_index=True)
    y, u, i, cut = (np.concatenate([p[j] for p in parts]) for j in range(1, 5))
    g = cut * n_users + u
    oof, rounds = cross_validate(F.to_numpy(), y, g, u, cfg["rounds"])
    on = cut == len(cfg["cuts"]) - 1
    m_rank = metrics(top_from_pairs(u[on], i[on], oof[on], n_users), targets)
    log("ranker", {"rounds": rounds, "rows": len(y)}, m_rank)

    full, ctx = generate(tr, n_users, E, ext, best, state, ft)
    Xf, allowed = ctx["X"], ctx["allowed"]
    uf, i_f = candidates(full, Xf, allowed, np.sort(te.userId.to_numpy()))
    Ff = features(tr, Xf, full, {"S": cosine(Xf), "E": E, "year": year, "ext": ext, "week": ctx["week"],
                                 "allowed": allowed}, uf, i_f)
    models = [fit(F.to_numpy(), y, g, rounds, seed=s) for s in cfg["rank_seeds"]]
    pred = np.mean([pd.Series(m.predict(Ff.to_numpy())).groupby(uf).rank(pct=True).to_numpy() for m in models], 0)
    tops = {name: recommend(s, Xf, allowed) for name, s in full.items()}
    tops["ranker"] = top_from_pairs(uf, i_f, pred, n_users)
    model = models[0]
    imp = pd.Series(model.feature_importance("gain"), index=F.columns).sort_values(ascending=False)
    print("ranker gain: " + ", ".join(f"{c} {v / imp.sum():.3f}" for c, v in imp.items()))

    best_single = max(single, key=lambda n: single[n][SELECT])
    winner = "ranker" if m_rank[SELECT] > single[best_single][SELECT] else best_single
    path = write(te, tops[winner], tops)
    print(f"submission model: {winner} (ranker {m_rank[SELECT]:.4f} vs best single {best_single} "
          f"{single[best_single][SELECT]:.4f})")
    return path


if __name__ == "__main__":
    tr, te, n_users = load()
    year, E = item_side()
    ext = external()
    run(SMOKE, tr, te, n_users, year, E, ext)
    print(f"smoke ok ({time.time() - T_START:.0f}s)", flush=True)
    if not os.environ.get("SMOKE_ONLY"):
        path = run(FULL, tr, te, n_users, year, E, ext)
        print(f"total {(time.time() - T_START) / 60:.1f} min")
        print(f"SUBMIT: {path}")
