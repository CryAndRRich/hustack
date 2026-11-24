import numpy as np
from sklearn.metrics import roc_auc_score
from config import N_BOOT, SEED

def auroc(y, score):
    if len(np.unique(y)) < 2:
        return float('nan')
    return float(roc_auc_score(y, score))

def precision_at_k(y, score, k):
    order = np.argsort(-score, kind='stable')[:k]
    return float(y[order].sum()) / float(k)

def recall_at_k(y, score, k):
    total = float(y.sum())
    if total == 0:
        return float('nan')
    order = np.argsort(-score, kind='stable')[:k]
    return float(y[order].sum()) / total

def report(y, score):
    return {'auroc': auroc(y, score), 'p50': precision_at_k(y, score, 50), 'r50': recall_at_k(y, score, 50), 'r100': recall_at_k(y, score, 100)}

def guide_bootstrap_wins(y, groups, score_a, score_b):
    rng = np.random.default_rng(SEED)
    guides = np.unique(groups)
    index_by_guide = {g: np.flatnonzero(groups == g) for g in guides}
    wins = 0
    valid = 0
    for _ in range(N_BOOT):
        picked = rng.choice(guides, size=len(guides), replace=True)
        idx = np.concatenate([index_by_guide[g] for g in picked])
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            continue
        valid += 1
        if roc_auc_score(yb, score_a[idx]) > roc_auc_score(yb, score_b[idx]):
            wins += 1
    return wins / valid if valid else float('nan')
