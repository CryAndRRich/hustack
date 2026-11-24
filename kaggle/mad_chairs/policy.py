import itertools

import numpy as np

from emulate import COVER, N, Tracker

OPTIONS = COVER
SKIP_ID = OPTIONS.index("skip")
WEAKEST_BONUS = 2.0
TARGET_WEIGHT = 4.0
OTHER_WEIGHT = 0.5
MIN_WEIGHT = 1e-6


def submission(position, round, history, cache):
    team = cache.setdefault("team", [])
    if round == 1:
        if position not in team:
            team.append(position)
        return COVER[position - 1]
    if cache.get("plan_round") != round:
        cache["plan_round"] = round
        cache["plan"] = plan(round, history, cache)
    return cache["plan"][position]


def _moves(history, k):
    return history.sort_values("Position")[f"Round{k}"].tolist()


def actions(k):
    return np.array(list(itertools.product(range(len(OPTIONS)), repeat=k)))


def plan(round, history, cache):
    team = sorted(p - 1 for p in cache["team"])
    opp = [p for p in range(N) if p not in team]
    if not opp:
        return {p + 1: COVER[(p + round - 1) % N] for p in team}
    tracker = cache.get("tracker")
    if tracker is None:
        tracker = cache["tracker"] = Tracker(opp)
    for k in range(tracker.state.round, round):
        tracker.observe(_moves(history, k))
    state = tracker.state
    scen = scenarios(tracker, opp)
    wins = state.wins
    weakest = min(team, key=lambda p: (wins[p], p))
    target = cache.get("target")
    if target is None or wins[target] > min(wins[p] for p in opp):
        target = min(opp, key=lambda p: (wins[p], p))
    cache["target"] = target
    ours_w = np.array([1.0 + WEAKEST_BONUS * (p == weakest) for p in team])
    theirs_w = np.array([OTHER_WEIGHT + TARGET_WEIGHT * (p == target) for p in opp])
    acts = actions(len(team))
    best = acts[int(np.argmax(values(acts, scen, ours_w, theirs_w)))]
    return {p + 1: OPTIONS[a] for p, a in zip(team, best)}


def scenarios(tracker, opp):
    merged = {}
    for h, w in tracker.posterior().items():
        if w < MIN_WEIGHT:
            continue
        dists = [tracker.state.dist(h, p) for p in opp]
        for combo in itertools.product(*[d.items() for d in dists]):
            key = tuple(OPTIONS.index(m) for m, _ in combo)
            merged[key] = merged.get(key, 0.0) + w * float(np.prod([q for _, q in combo]))
    keys = list(merged)
    return np.array(keys), np.array([merged[k] for k in keys])


def values(acts, scen, ours_w, theirs_w):
    moves, weights = scen
    k = acts.shape[1]
    full = np.concatenate([np.repeat(acts[:, None, :], len(moves), 1),
                           np.repeat(moves[None, :, :], len(acts), 0)], axis=2)
    unique = (full[..., :, None] == full[..., None, :]).sum(-1) == 1
    won = unique & (full != SKIP_ID)
    v = won[..., :k] @ ours_w - won[..., k:] @ theirs_w
    return v @ weights
