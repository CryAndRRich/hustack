from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable

from nlp.data import Block, Document
from nlp.question import Condition, Query
from nlp.table import Table, find_table

FALLBACK_ANSWER = "0"
NUMBER = re.compile(r"[+-]?\d[\d.,]*")
STAMP = " Đã đối chiếu"


def is_noisy(block: Block | None) -> bool:
    return block is not None and block.text.endswith(STAMP) and block.text != STAMP.strip()


BoldScorer = Callable[[Document, Table, int], float]


@dataclass
class Answer:
    answer: str
    evidence: list[Block] = field(default_factory=list)
    solved: bool = True

    def to_json(self, question_id: str) -> dict:
        seen, evidence = set(), []
        for block in self.evidence:
            if block is not None and block.block_id not in seen:
                seen.add(block.block_id)
                evidence.append(block.evidence)
        return {"question_id": question_id, "answer": self.answer, "evidence": evidence}


def parse_number(text: str) -> tuple[float, int] | None:
    match = NUMBER.match(text.strip().replace(" ", "") if NUMBER.fullmatch(text.strip().replace(" ", "")) else text.strip())
    if not match:
        return None
    token = match.group(0).rstrip(".,")
    decimals = 0
    if "," in token:
        whole, _, frac = token.rpartition(",")
        decimals = len(frac)
        token = whole.replace(".", "") + "." + frac
    else:
        token = token.replace(".", "")
    try:
        return float(token), decimals
    except ValueError:
        return None


def format_number(value: float, decimals: int) -> str:
    if decimals == 0 or abs(value - round(value)) < 1e-9 and decimals == 0:
        return str(int(round(value)))
    text = f"{value:.{decimals}f}"
    return text.replace(".", ",")


def anchored(table: Table, row: int, col: int) -> Block | None:
    block = table.cell(row, col)
    return block if block is not None and block.row0 == row and block.col0 == col else None


def column_cell(table: Table, row: int, name: str) -> Block | None:
    for col in table.column_index(name):
        block = anchored(table, row, col)
        if block is not None:
            return block
    return None


def condition_cells(table: Table, row: int, conditions: list[Condition]) -> list[Block] | None:
    cells = []
    for header, value in conditions:
        block = column_cell(table, row, header)
        if block is None or block.text != value:
            return None
        cells.append(block)
    return cells


def matching_rows(table: Table, conditions: list[Condition]) -> list[tuple[int, list[Block]]]:
    result = []
    for row in table.data_rows:
        cells = condition_cells(table, row, conditions)
        if cells is not None:
            result.append((row, cells))
    return result


def split_two_rows(table: Table, conditions: list[Condition]):
    for cut in range(1, len(conditions)):
        first, second = matching_rows(table, conditions[:cut]), matching_rows(table, conditions[cut:])
        if len(first) == 1 and len(second) == 1 and first[0][0] != second[0][0]:
            return first[0], second[0]
    return None


def key_width(table: Table, row: int) -> int:
    rows = table.data_rows

    def key(r: int, k: int) -> tuple:
        return tuple(b.text if (b := anchored(table, r, c)) is not None else None for c in range(k))

    for k in range(1, table.n_cols + 1):
        if sum(1 for r in rows if key(r, k) == key(row, k)) == 1:
            return k
    return 1


def key_columns(table: Table, row: int) -> list[Block]:
    return [b for b in (anchored(table, row, c) for c in range(key_width(table, row))) if b is not None]


def identifiable(table: Table, row: int, target_col: int) -> bool:
    k = key_width(table, row)
    cells = [anchored(table, row, c) for c in range(k)]
    return target_col >= k and all(c is not None and not is_noisy(c) for c in cells)


def solve_lookup(doc: Document, q: Query, **_) -> Answer | None:
    table = find_table(doc, q.page, q.table)
    rows = matching_rows(table, q.conditions) if table else []
    for row, cells in rows:
        target = column_cell(table, row, q.target)
        if target is not None:
            return Answer(target.text, cells + [target])
    return None


def _two_rows(table: Table, q: Query):
    if q.rows:
        first, second = matching_rows(table, q.rows[0]), matching_rows(table, q.rows[1])
        return (first[0], second[0]) if first and second else None
    return split_two_rows(table, q.conditions)


def solve_sum(doc: Document, q: Query, **_) -> Answer | None:
    table = find_table(doc, q.page, q.table)
    pair = _two_rows(table, q) if table else None
    if pair is None:
        return None
    targets = [column_cell(table, row, q.target) for row, _ in pair]
    values = [parse_number(t.text) if t else None for t in targets]
    if any(v is None for v in values):
        return None
    total = sum(v for v, _ in values)
    return Answer(format_number(total, max(d for _, d in values)), pair[0][1] + pair[1][1] + targets)


def solve_extreme(doc: Document, q: Query, **_) -> Answer | None:
    table = find_table(doc, q.page, q.table)
    if table is None:
        return None
    label_counts = Counter(b.text for b in (column_cell(table, r, q.label) for r in table.data_rows) if b)
    best = None
    for row in table.data_rows:
        target, label = column_cell(table, row, q.target), column_cell(table, row, q.label)
        if target is None or label is None or is_noisy(target) or is_noisy(label):
            continue
        if label_counts[label.text] > 1 and not identifiable(table, row, target.col0):
            continue
        value = parse_number(target.text)
        if value is None:
            continue
        better = best is None or (value[0] > best[0] if q.kind == "argmax" else value[0] < best[0])
        if better:
            best = (value[0], row, label, target)
    if best is None:
        return None
    _, row, label, target = best
    return Answer(label.text, [label] + key_columns(table, row) + [target])


def solve_compare(doc: Document, q: Query, **_) -> Answer | None:
    table = find_table(doc, q.page, q.table)
    pair = _two_rows(table, q) if table else None
    if pair is None:
        return None
    targets = [column_cell(table, row, q.target) for row, _ in pair]
    values = [parse_number(t.text) if t else None for t in targets]
    if any(v is None for v in values):
        return None
    winner = 0 if values[0][0] >= values[1][0] else 1
    label = anchored(table, pair[winner][0], 0)
    if label is None:
        return None
    return Answer(label.text, pair[0][1] + pair[1][1] + targets)


def solve_count(doc: Document, q: Query, **_) -> Answer | None:
    table = find_table(doc, q.page, q.table)
    if table is None:
        return None
    rows = matching_rows(table, q.conditions)
    return Answer(str(len(rows)), [cells[0] for _, cells in rows])


def solve_cross_page_sum(doc: Document, q: Query, **_) -> Answer | None:
    parts = []
    for page, conditions, target_name in zip(q.pages, q.rows, q.targets):
        found = None
        for table in (t for t in doc.tables if t.page == page):
            rows = matching_rows(table, conditions)
            if rows and table.column_index(target_name):
                row, cells = rows[0]
                target = column_cell(table, row, target_name)
                if target is not None:
                    found = (cells, target)
                    break
        if found is None:
            return None
        parts.append(found)
    values = [parse_number(target.text) for _, target in parts]
    if any(v is None for v in values):
        return None
    total = sum(v for v, _ in values)
    evidence = [b for cells, target in parts for b in cells + [target]]
    return Answer(format_number(total, max(d for _, d in values)), evidence)


def solve_bold(doc: Document, q: Query, bold_scorer: BoldScorer | None = None, **_) -> Answer | None:
    table = find_table(doc, q.page, q.table)
    pair = _two_rows(table, q) if table else None
    if pair is None:
        return None
    scores = [bold_scorer(doc, table, row) if bold_scorer else 0.5 for row, _ in pair]
    bold = 0 if scores[0] >= scores[1] else 1
    target = column_cell(table, pair[bold][0], q.target)
    if target is None:
        target = column_cell(table, pair[1 - bold][0], q.target)
    if target is None:
        return None
    return Answer(target.text, pair[0][1] + pair[1][1] + [target])


SOLVERS = {
    "lookup": solve_lookup,
    "sum": solve_sum,
    "argmax": solve_extreme,
    "argmin": solve_extreme,
    "compare": solve_compare,
    "count": solve_count,
    "cross_page_sum": solve_cross_page_sum,
    "visual_bold_lookup": solve_bold,
}


def solve(doc: Document, q: Query, bold_scorer: BoldScorer | None = None) -> Answer:
    solver = SOLVERS.get(q.kind)
    try:
        result = solver(doc, q, bold_scorer=bold_scorer) if solver else None
    except (KeyError, IndexError, ValueError):
        result = None
    return result if result is not None else Answer(FALLBACK_ANSWER, [], solved=False)
