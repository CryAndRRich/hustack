from collections import Counter, defaultdict

from emulate import DETERMINISTIC, Match, Tracker

ROUNDS = 20
CHALLENGER = "challenger"


def match_moves(rows):
    rows = rows.sort_values("Position")
    return [rows[f"Round{k}"].tolist() for k in range(1, ROUNDS + 1)], rows["Strategy"].tolist()


def replay(results, log):
    bad = Counter()
    checked = Counter()
    examples = []
    ident = defaultdict(Counter)
    settle = defaultdict(list)
    for match, rows in results.groupby("Match", sort=False):
        moves, strats = match_moves(rows)
        state = Match()
        for r, mv in enumerate(moves, 1):
            for p, s in enumerate(strats):
                if s in DETERMINISTIC or s == "random3":
                    checked[s] += 1
                    ok = state.move(s, p) == mv[p] if s in DETERMINISTIC else state.dist(s, p).get(mv[p], 0) > 0
                    if not ok:
                        bad[s] += 1
                        if len(examples) < 10:
                            examples.append((match, r, p + 1, s, mv[p]))
            state.observe(mv)
        opp = [p for p, s in enumerate(strats) if s != CHALLENGER]
        if CHALLENGER in strats and opp:
            truth = strats[opp[0]]
            tracker = Tracker(opp)
            first_right = None
            for r, mv in enumerate(moves, 1):
                tracker.observe(mv)
                post = tracker.posterior()
                right = max(post, key=post.get) == truth
                first_right = (first_right or r) if right else None
            ident[truth][max(post, key=post.get)] += 1
            settle[truth].append(first_right)
    log("emulator check (mismatches / moves):",
        {s: f"{bad[s]}/{checked[s]}" for s in sorted(checked)})
    for e in examples:
        log("  mismatch match/round/position/strategy/move:", e)
    for truth in sorted(ident):
        rounds = [x for x in settle[truth] if x is not None]
        log(f"  identification {truth}: final MAP {dict(ident[truth])}; settled by round "
            f"{max(rounds) if rounds else None} in {len(rounds)}/{len(settle[truth])}")
    return {"mismatch": dict(bad), "checked": dict(checked),
            "identification": {k: dict(v) for k, v in ident.items()}}


def challenger_table(results):
    out = {}
    for match, rows in results.groupby("Match", sort=False):
        strats = rows["Strategy"].tolist()
        if CHALLENGER not in strats:
            continue
        opp = next((s for s in strats if s != CHALLENGER), "self")
        t = out.setdefault(opp, {"win": 0, "draw": 0, "loss": 0, "our_min": 0.0, "their_min": 0.0,
                                 "our_mean": 0.0, "their_mean": 0.0, "n": 0})
        winner = rows["Winner"].iloc[0]
        t["win" if winner == CHALLENGER else "draw" if winner == "draw" else "loss"] += 1
        ours = rows.loc[rows["Strategy"] == CHALLENGER, "Win"]
        theirs = rows.loc[rows["Strategy"] != CHALLENGER, "Win"] if opp != "self" else ours
        t["our_min"] += ours.min()
        t["their_min"] += theirs.min()
        t["our_mean"] += ours.mean()
        t["their_mean"] += theirs.mean()
        t["n"] += 1
    for t in out.values():
        for k in ("our_min", "their_min", "our_mean", "their_mean"):
            t[k] = round(t[k] / t["n"], 4)
    return out
