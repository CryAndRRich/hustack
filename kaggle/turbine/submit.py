import numpy as np
import pandas as pd

import config


def top_ids(ids, score, k, forced=None):
    if forced is None or not forced.any():
        pick = np.argsort(-score, kind="stable")[:k]
        return ids[pick]
    rest = np.flatnonzero(~forced)
    fill = rest[np.argsort(-score[rest], kind="stable")[: k - int(forced.sum())]]
    return np.concatenate([ids[forced], ids[fill]])


def write(path, chosen, all_ids, k):
    chosen = np.asarray(chosen)
    if len(chosen) != k or len(set(chosen)) != k:
        raise ValueError("submission must hold exactly %d unique ids, got %d (%d unique)"
                         % (k, len(chosen), len(set(chosen))))
    universe = set(all_ids)
    if any(c not in universe for c in chosen):
        raise ValueError("submission contains ids outside the candidates")
    pd.DataFrame({"hashed_aaid": chosen, "prediction": 0}).to_csv(path, index=False)


def forced_mask(cand):
    return cand["ad_revenue"] > 0


def k_for(cand):
    return min(config.K, len(cand["ids"]))
