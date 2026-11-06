from __future__ import annotations

from collections import defaultdict

from cv.catalog import STEP_HOURS, parse_time

EPS = 1e-6


def matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [[sum(a[i][k] * b[k][j] for k in range(len(b))) for j in range(len(b[0]))] for i in range(len(a))]


def matrix_powers(a: list[list[float]], n: int) -> list[list[list[float]]]:
    k = len(a)
    out = [[[float(i == j) for j in range(k)] for i in range(k)]]
    for _ in range(n):
        out.append(matmul(out[-1], a))
    return out


def gap_steps(t0: str, t1: str) -> int:
    return round((parse_time(t1) - parse_time(t0)).total_seconds() / 3600 / STEP_HOURS)


def sequences(keys: list[tuple[str, str]], max_gap_steps: int) -> list[list[tuple[int, int]]]:
    by_storm = defaultdict(list)
    for i, (storm, stamp) in enumerate(keys):
        by_storm[storm].append((stamp, i))
    out = []
    for storm in sorted(by_storm):
        items = sorted(by_storm[storm])
        current = [(items[0][1], 0)]
        for (prev, _), (stamp, i) in zip(items, items[1:]):
            gap = gap_steps(prev, stamp)
            if gap > max_gap_steps:
                out.append(current)
                current = [(i, 0)]
            else:
                current.append((i, gap))
        out.append(current)
    return out


def transition_matrix(keys: list[tuple[str, str]], labels: list[int], k: int):
    counts = [[1.0] * k for _ in range(k)]
    for seq in sequences(keys, max_gap_steps=1):
        for (a, _), (b, gap) in zip(seq, seq[1:]):
            if gap == 1:
                counts[labels[a]][labels[b]] += 1
    return [[c / sum(row) for c in row] for row in counts]


def class_prior(labels: list[int], k: int) -> list[float]:
    counts = [1.0] * k
    for y in labels:
        counts[y] += 1
    return [c / sum(counts) for c in counts]


def _normalize(v: list[float]) -> list[float]:
    s = sum(v)
    return [x / s for x in v] if s > 0 else [1.0 / len(v)] * len(v)


def forward_backward(emissions: list[list[float]], gaps: list[int], powers, init: list[float]) -> list[list[float]]:
    k = len(init)
    n = len(emissions)
    alpha = [_normalize([init[j] * emissions[0][j] for j in range(k)])]
    for t in range(1, n):
        p = powers[gaps[t]]
        prev = alpha[-1]
        alpha.append(_normalize([emissions[t][j] * sum(prev[i] * p[i][j] for i in range(k)) for j in range(k)]))
    beta = [[1.0] * k for _ in range(n)]
    for t in range(n - 2, -1, -1):
        p = powers[gaps[t + 1]]
        nxt = [emissions[t + 1][j] * beta[t + 1][j] for j in range(k)]
        beta[t] = _normalize([sum(p[i][j] * nxt[j] for j in range(k)) for i in range(k)])
    return [_normalize([a * b for a, b in zip(alpha[t], beta[t])]) for t in range(n)]


def smooth(probs: list[list[float]], keys: list[tuple[str, str]], transition, prior: list[float],
           alpha: float, mix: float, max_gap_steps: int, gamma: float = 1.0) -> list[list[float]]:
    k = len(prior)
    a = [[(1 - mix) * transition[i][j] + mix / k for j in range(k)] for i in range(k)]
    powers = matrix_powers(a, max_gap_steps)
    out: list[list[float] | None] = [None] * len(probs)
    for seq in sequences(keys, max_gap_steps):
        emissions = [[(max(probs[i][j], EPS) / prior[j] ** gamma) ** alpha for j in range(k)] for i, _ in seq]
        posterior = forward_backward(emissions, [g for _, g in seq], powers, prior)
        for (i, _), post in zip(seq, posterior):
            out[i] = post
    return out


def accuracy(probs: list[list[float]], labels: list[int]) -> float:
    return sum(max(range(len(p)), key=p.__getitem__) == y for p, y in zip(probs, labels)) / max(1, len(labels))


def tune(probs, keys, labels, transition, prior, alphas, mixes, gammas, max_gap_steps) -> tuple[dict, list[dict]]:
    table = []
    for gamma in gammas:
        for alpha in alphas:
            for mix in mixes:
                acc = accuracy(smooth(probs, keys, transition, prior, alpha, mix, max_gap_steps, gamma), labels)
                table.append({"alpha": alpha, "mix": mix, "gamma": gamma, "acc": acc})
    best = max(table, key=lambda r: (round(r["acc"], 6), r["gamma"], r["mix"], r["alpha"]))
    return best, table
