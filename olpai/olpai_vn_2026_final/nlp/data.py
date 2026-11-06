from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from nlp.config import DOMAINS

Triple = tuple[str, str, str]


@dataclass
class Example:
    domain: str
    index: int
    raw_words: str
    triples: list[Triple] = field(default_factory=list)

    @property
    def tokens(self) -> list[str]:
        return self.raw_words.split()


def normalize_aspect(aspect) -> str:
    if aspect is None:
        return "NULL"
    if isinstance(aspect, list):
        if not aspect or (len(aspect) == 1 and aspect[0] in ("NULL", "null", "")):
            return "NULL"
        return " ".join(str(t).strip() for t in aspect).strip()
    text = str(aspect).strip()
    return "NULL" if text.upper() == "NULL" or text == "" else text


def gold_triples(quadruples: list[dict]) -> list[Triple]:
    keyed = {}
    for order, q in enumerate(quadruples):
        triple = (normalize_aspect(q["aspect"].get("term")), q["category"].strip(), q["sentiment"].strip().upper())
        start = q["aspect"].get("from", -1)
        position = start if triple[0] != "NULL" and start is not None and start >= 0 else 10**6
        keyed.setdefault(triple, (position, order))
    return sorted(keyed, key=keyed.get)


def split_path(data_root: Path, name: str) -> Path:
    candidates = {
        "Train": [data_root / "train" / "Train.json"],
        "Dev": [data_root / "train" / "Dev.json"],
        "PublicTest": [data_root / "public_test" / "PublicTest.json"],
        "PrivateTest": [data_root / "private_test" / "PrivateTest.json", data_root / "PrivateTest.json"],
    }[name]
    for path in candidates:
        if path.is_file():
            return path
    hint = " (private_test.zip not unzipped?)" if name == "PrivateTest" else ""
    raise FileNotFoundError(f"{name}.json not found in {[str(p) for p in candidates]}{hint}")


def test_split_name(split: str) -> str:
    return {"public_test": "PublicTest", "private_test": "PrivateTest"}[split]


def load_examples(path: Path) -> list[Example]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    missing = [d for d in DOMAINS if d not in data]
    if missing:
        raise KeyError(f"{path} missing domains {missing}")
    examples = []
    for domain in DOMAINS:
        for i, item in enumerate(data[domain]):
            triples = gold_triples(item["quadruples"]) if "quadruples" in item else []
            examples.append(Example(domain, i, item["raw_words"], triples))
    return examples


def category_sets(*example_lists: list[Example]) -> dict[str, list[str]]:
    sets = {d: set() for d in DOMAINS}
    for examples in example_lists:
        for ex in examples:
            sets[ex.domain].update(c for _, c, _ in ex.triples)
    return {d: sorted(s) for d, s in sets.items()}


def to_submission(examples: list[Example], predictions: dict[tuple[str, int], list[Triple]]) -> dict:
    out = {d: [] for d in DOMAINS}
    for ex in sorted(examples, key=lambda e: (DOMAINS.index(e.domain), e.index)):
        triples = predictions.get((ex.domain, ex.index), [])
        out[ex.domain].append({
            "raw_words": ex.raw_words,
            "triples": [{"aspect": a, "category": c, "sentiment": s} for a, c, s in triples],
        })
    return out
