import numpy as np


def capture_sorted(rev_sorted, w_sorted, frac):
    cw = np.cumsum(w_sorted)
    take = cw <= frac * cw[-1]
    rw = rev_sorted * w_sorted
    return rw[take].sum() / rw.sum()


def capture(score, rev, w, frac):
    order = np.argsort(-score, kind="stable")
    return capture_sorted(rev[order], w[order], frac)


def table(scores, rev, w, frac):
    rows = {arm: capture(s, rev, w, frac) for arm, s in scores.items()}
    rows["oracle"] = capture(rev, rev, w, frac)
    rows["random"] = frac
    return rows


def gate(scores, rev, w, frac, candidate, n_boot, win, seed):
    rng = np.random.default_rng(seed)
    orders = {arm: np.argsort(-s, kind="stable") for arm, s in scores.items()}
    wins = {arm: 0 for arm in scores if arm != candidate}
    for _ in range(n_boot):
        m = rng.poisson(1.0, len(rev)).astype(np.float64)
        ww = w * m
        base = capture_sorted(rev[orders[candidate]], ww[orders[candidate]], frac)
        for arm in wins:
            o = orders[arm]
            if capture_sorted(rev[o], ww[o], frac) > base:
                wins[arm] += 1
    share = {arm: n / n_boot for arm, n in wins.items()}
    best = max(share, key=share.get) if share else None
    chosen = best if best is not None and share[best] >= win else candidate
    return chosen, share
