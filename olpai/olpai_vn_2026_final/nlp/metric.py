from __future__ import annotations

from nlp.config import DOMAINS
from nlp.data import Example, Triple, normalize_aspect

SENTIMENTS = ("POS", "NEG", "NEU")


def _normalize(triple) -> Triple:
    aspect, category, sentiment = triple
    return normalize_aspect(aspect), (category or "").strip(), (sentiment or "").strip().upper()


def evaluate(examples: list[Example], predictions: dict[tuple[str, int], list[Triple]]) -> dict:
    counts = {d: [0, 0, 0] for d in DOMAINS}
    for ex in examples:
        gold = {_normalize(t) for t in ex.triples}
        pred = {_normalize(t) for t in predictions.get((ex.domain, ex.index), [])}
        c = counts[ex.domain]
        c[0] += len(gold & pred)
        c[1] += len(pred)
        c[2] += len(gold)
    result = {}
    for d, (correct, n_pred, n_gold) in counts.items():
        p = correct / n_pred if n_pred else 0.0
        r = correct / n_gold if n_gold else 0.0
        f1 = 2 * p * r / (p + r) if p + r > 0 else 0.0
        result[d] = {"precision": p, "recall": r, "f1": f1, "gold": n_gold, "pred": n_pred, "correct": correct}
    present = [d for d in DOMAINS if counts[d][2] or counts[d][1]]
    result["avg_f1"] = sum(result[d]["f1"] for d in present) / max(1, len(present))
    return result


def format_scores(scores: dict) -> str:
    rows = [f"  {d:<10} P={scores[d]['precision']:.4f} R={scores[d]['recall']:.4f} F1={scores[d]['f1']:.4f} "
            f"(gold {scores[d]['gold']}, pred {scores[d]['pred']}, correct {scores[d]['correct']})"
            for d in DOMAINS if d in scores]
    return "\n".join(rows + [f"  {'mean 5 dom':<10} F1={scores['avg_f1']:.4f}"])


def validate_submission(submission: dict, test_examples: list[Example]) -> None:
    if list(submission) != list(DOMAINS):
        raise ValueError(f"Wrong domain keys: {list(submission)}")
    by_domain = {d: [ex for ex in test_examples if ex.domain == d] for d in DOMAINS}
    for d in DOMAINS:
        items, expected = submission[d], sorted(by_domain[d], key=lambda e: e.index)
        if len(items) != len(expected):
            raise ValueError(f"{d}: {len(items)} sentences, test has {len(expected)}")
        for item, ex in zip(items, expected):
            if set(item) != {"raw_words", "triples"} or item["raw_words"] != ex.raw_words:
                raise ValueError(f"{d}[{ex.index}]: wrong keys or raw_words differ from test")
            for t in item["triples"]:
                if set(t) != {"aspect", "category", "sentiment"} or t["sentiment"] not in SENTIMENTS:
                    raise ValueError(f"{d}[{ex.index}]: malformed triple {t}")
                if not t["aspect"] or not t["category"]:
                    raise ValueError(f"{d}[{ex.index}]: empty aspect/category {t}")
