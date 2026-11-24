import itertools

import numpy as np

from metric import decide, score


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def blend(preds, w):
    return 1 / (1 + np.exp(-sum(wi * logit(p) for wi, p in zip(w, preds) if wi > 0)))


def simplex(n, step):
    k = int(round(1 / step))
    return [np.array(c) / k for c in itertools.product(range(k + 1), repeat=n) if sum(c) == k]


def moment_table(p, window, y, pi, safeties, weight=None):
    return {s: score(decide(p, window, pi, s), y, window, weight)[1] for s in safeties}


def evaluate(cfg, w, sets):
    tables = [moment_table(blend(s["preds"], w), s["window"], s["y"], s["pi"], cfg.safety, s.get("weight"))
              for s in sets]
    moments = sorted(set().union(*[t[cfg.safety[0]].keys() for t in tables]))
    safety, per = {}, {}
    for m in moments:
        combined = {s: np.mean([t[s][m] for t in tables if m in t[s]]) for s in cfg.safety}
        safety[m] = max(combined, key=combined.get)
        per[m] = combined[safety[m]]
    return float(np.mean(list(per.values()))), safety, per


def search(cfg, names, sets):
    results = []
    for w in simplex(len(names), cfg.blend_step):
        total, safety, per = evaluate(cfg, w, sets)
        results.append((total, w, safety, per))
    results.sort(key=lambda r: -r[0])
    single = {}
    for i, name in enumerate(names):
        w = np.eye(len(names))[i]
        single[name] = evaluate(cfg, w, sets)[0]
    return results[0], single
