import numpy as np

STABLE, UNSTABLE, WAIT = "stable", "unstable", "wait"


def gain(window_ms):
    return 1 - np.asarray(window_ms, dtype=float) / 600


def score(decision, unstable, window_ms, weight=None):
    decision, unstable, window_ms = np.asarray(decision), np.asarray(unstable), np.asarray(window_ms)
    weight = np.ones(len(decision)) if weight is None else np.asarray(weight, dtype=float)
    g = gain(window_ms)
    truth = np.where(unstable == 1, UNSTABLE, STABLE)
    pts = np.where(decision == WAIT, 0.0, np.where(decision == truth, g, -2.0))
    per = {}
    for w in np.unique(window_ms):
        m = window_ms == w
        halves = [np.average(pts[m & (unstable == c)], weights=weight[m & (unstable == c)])
                  for c in (0, 1) if (m & (unstable == c)).any()]
        per[int(w)] = 100 * np.mean(halves) / gain(w)
    return float(np.mean(list(per.values()))), per


def decide(p, window_ms, pi_unstable, safety=1.0, no_stable=None):
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1 - 1e-9)
    g = gain(window_ms)
    ratio = (1 - pi_unstable) / pi_unstable
    call_u = p / (1 - p) > safety * 2 / (g * ratio)
    call_s = (1 - p) / p > safety * 2 * ratio / g
    if no_stable is not None:
        call_s &= ~np.asarray(no_stable, dtype=bool)
    return np.where(call_u, UNSTABLE, np.where(call_s, STABLE, WAIT))
