from __future__ import annotations

import difflib

from nlp.data import Example, Triple

TRIPLE_SEP = " [SSEP] "
FIELD_SEP = " | "
EMPTY = "no triples"
SENTIMENT_WORDS = {"POS": "positive", "NEG": "negative", "NEU": "neutral"}
SENTIMENT_CODES = {**{v: k for k, v in SENTIMENT_WORDS.items()}, **{k.lower(): k for k in SENTIMENT_WORDS}}


def verbalize_category(category: str) -> str:
    return category.replace("#", " ").replace("_", " ").lower()


def source_text(ex: Example, causal: bool) -> str:
    text = f"Domain: {ex.domain}. Review: {ex.raw_words}"
    return text + "\nTriples:" if causal else text


def target_text(triples: list[Triple], verbalize: bool) -> str:
    if not triples:
        return EMPTY
    parts = []
    for aspect, category, sentiment in triples:
        if verbalize:
            category, sentiment = verbalize_category(category), SENTIMENT_WORDS[sentiment]
        parts.append(FIELD_SEP.join((aspect, category, sentiment)))
    return TRIPLE_SEP.join(parts)


class CategoryMapper:
    def __init__(self, categories: dict[str, list[str]]):
        self.lookup = {}
        self.choices = {}
        for domain, cats in categories.items():
            table = {}
            for c in cats:
                table[c.lower()] = c
                table[verbalize_category(c)] = c
            self.lookup[domain] = table
            self.choices[domain] = sorted(table)

    def __call__(self, domain: str, text: str) -> str | None:
        key = text.strip().lower()
        table = self.lookup[domain]
        if key in table:
            return table[key]
        close = difflib.get_close_matches(key, self.choices[domain], n=1, cutoff=0.8)
        return table[close[0]] if close else None


def align_aspect(text: str, tokens: list[str], threshold: float) -> str | None:
    text = text.strip()
    if not text or text.upper() == "NULL":
        return "NULL"
    sentence = " ".join(tokens)
    if f" {text} " in f" {sentence} ":
        return text
    n = len(text.split())
    key = text.replace(" ", "").lower()
    lengths = range(1, min(len(tokens), n + 3) + 1)
    for length in lengths:
        for i in range(len(tokens) - length + 1):
            if "".join(tokens[i:i + length]).lower() == key:
                return " ".join(tokens[i:i + length])
    if threshold >= 1:
        return None
    best, best_ratio = None, 0.0
    for length in range(max(1, n - 2), min(len(tokens), n + 2) + 1):
        for i in range(len(tokens) - length + 1):
            span = " ".join(tokens[i:i + length])
            ratio = difflib.SequenceMatcher(None, span.lower(), text.lower()).ratio()
            if ratio > best_ratio:
                best, best_ratio = span, ratio
    return best if best_ratio >= threshold else None


def parse_output(text: str, ex: Example, mapper: CategoryMapper, aspect_fuzzy: float = 0.8,
                 drop_unknown_category: bool = True) -> list[Triple]:
    triples, seen = [], set()
    text = text.strip()
    if not text or text.lower() == EMPTY:
        return []
    tokens = ex.tokens
    for chunk in text.split(TRIPLE_SEP.strip()):
        parts = [p.strip() for p in chunk.split(FIELD_SEP.strip())]
        if len(parts) != 3:
            continue
        aspect_text, category_text, sentiment_text = parts
        sentiment = SENTIMENT_CODES.get(sentiment_text.lower())
        category = mapper(ex.domain, category_text)
        if category is None and not drop_unknown_category:
            category = category_text
        aspect = align_aspect(aspect_text, tokens, aspect_fuzzy)
        if sentiment is None or category is None or aspect is None:
            continue
        triple = (aspect, category, sentiment)
        if triple not in seen:
            seen.add(triple)
            triples.append(triple)
    return triples
