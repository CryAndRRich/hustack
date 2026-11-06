from __future__ import annotations

ANLS_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
ANSWER_WEIGHT, EVIDENCE_WEIGHT = 0.85, 0.15


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def anls(prediction: str, answers: list[str]) -> float:
    pred = str(prediction).strip().lower()
    best = 0.0
    for answer in answers:
        gold = str(answer).strip().lower()
        length = max(len(pred), len(gold))
        nl = levenshtein(pred, gold) / length if length else 0.0
        best = max(best, 1 - nl if nl < ANLS_THRESHOLD else 0.0)
    return best


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def evidence_f1(predicted: list[dict], gold: list[dict]) -> float:
    if not predicted and not gold:
        return 1.0
    if not predicted or not gold:
        return 0.0
    pairs = sorted(
        ((iou(p["bbox"], g["bbox"]), i, j)
         for i, p in enumerate(predicted) for j, g in enumerate(gold) if int(p["page"]) == int(g["page"])),
        reverse=True,
    )
    used_p, used_g, matched = set(), set(), 0
    for value, i, j in pairs:
        if value < IOU_THRESHOLD:
            break
        if i not in used_p and j not in used_g:
            used_p.add(i)
            used_g.add(j)
            matched += 1
    if matched == 0:
        return 0.0
    precision, recall = matched / len(predicted), matched / len(gold)
    return 2 * precision * recall / (precision + recall)


def question_score(prediction: dict, label: dict) -> dict:
    a = anls(prediction["answer"], label["answers"])
    e = evidence_f1(prediction["evidence"], label["evidence"])
    return {"anls": a, "evidence_f1": e, "score": ANSWER_WEIGHT * a + EVIDENCE_WEIGHT * e}


def evaluate(predictions: dict[str, dict], labels: dict[str, dict]) -> dict:
    by_type: dict[str, list[dict]] = {}
    for qid, label in labels.items():
        result = question_score(predictions[qid], label)
        by_type.setdefault(label["reasoning_type"], []).append(result)
        by_type.setdefault("ALL", []).append(result)

    def mean(rows, key):
        return 100 * sum(r[key] for r in rows) / len(rows)

    return {kind: {"n": len(rows), "score": mean(rows, "score"), "anls": mean(rows, "anls"),
                   "evidence_f1": mean(rows, "evidence_f1")} for kind, rows in by_type.items()}


def validate_predictions(rows: list[dict], expected_ids: list[str], page_counts: dict[str, int]) -> None:
    ids = [r["question_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate question_id")
    if set(ids) != set(expected_ids):
        raise ValueError(f"Question set mismatch: {len(set(expected_ids) - set(ids))} missing, {len(set(ids) - set(expected_ids))} extra")
    for r in rows:
        if set(r) != {"question_id", "answer", "evidence"}:
            raise ValueError(f"{r['question_id']}: invalid keys {sorted(r)}")
        if not isinstance(r["answer"], str) or not r["answer"].strip():
            raise ValueError(f"{r['question_id']}: empty answer")
        doc_pages = page_counts[r["question_id"].rsplit("-q", 1)[0]]
        for e in r["evidence"]:
            if set(e) != {"page", "bbox"} or not isinstance(e["page"], int) or not 1 <= e["page"] <= doc_pages:
                raise ValueError(f"{r['question_id']}: invalid evidence {e}")
            x1, y1, x2, y2 = e["bbox"]
            if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
                raise ValueError(f"{r['question_id']}: invalid bbox {e['bbox']}")
