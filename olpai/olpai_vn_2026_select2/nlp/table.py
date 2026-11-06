from __future__ import annotations

from dataclasses import dataclass, field

from nlp.data import Block, Document

EDGE_TOL = 1e-4


def _cluster(values: list[float]) -> list[float]:
    edges: list[float] = []
    for v in sorted(values):
        if not edges or v - edges[-1] > EDGE_TOL:
            edges.append(v)
    return edges


def _index(edges: list[float], value: float) -> int:
    return min(range(len(edges)), key=lambda i: abs(edges[i] - value))


@dataclass
class Table:
    page: int
    index: int
    blocks: list[Block]
    n_rows: int
    n_cols: int
    header_rows: int
    title: Block | None
    columns: list[str] = field(default_factory=list)
    grid: dict[tuple[int, int], Block] = field(default_factory=dict)

    def cell(self, row: int, col: int) -> Block | None:
        return self.grid.get((row, col))

    @property
    def data_rows(self) -> list[int]:
        return list(range(self.header_rows, self.n_rows))

    def column_index(self, name: str) -> list[int]:
        return [c for c, n in enumerate(self.columns) if n == name]


def _segment(blocks: list[Block]) -> list[list[Block]]:
    ordered = sorted(blocks, key=lambda b: b.bbox[1])
    groups: list[list[Block]] = []
    bottom = -1.0
    for block in ordered:
        if not groups or block.bbox[1] > bottom + EDGE_TOL:
            groups.append([])
            bottom = block.bbox[3]
        groups[-1].append(block)
        bottom = max(bottom, block.bbox[3])
    return groups


def _build_table(page: int, index: int, blocks: list[Block]) -> Table:
    xs = _cluster([v for b in blocks for v in (b.bbox[0], b.bbox[2])])
    ys = _cluster([v for b in blocks for v in (b.bbox[1], b.bbox[3])])
    for b in blocks:
        b.table = index
        b.col0, b.col1 = _index(xs, b.bbox[0]), _index(xs, b.bbox[2])
        b.row0, b.row1 = _index(ys, b.bbox[1]), _index(ys, b.bbox[3])
    n_rows, n_cols = len(ys) - 1, len(xs) - 1

    first_row = [b for b in blocks if b.row0 == 0]
    title = first_row[0] if len(first_row) == 1 and first_row[0].col0 == 0 and first_row[0].col1 == n_cols else None
    header_rows = 3 if title is not None else 1

    table = Table(page, index, blocks, n_rows, n_cols, header_rows, title)
    for b in blocks:
        for r in range(b.row0, b.row1):
            for c in range(b.col0, b.col1):
                table.grid[(r, c)] = b
    columns = []
    for c in range(n_cols):
        headers = [table.grid.get((r, c)) for r in range(1 if title else 0, header_rows)]
        headers = [h for h in headers if h is not None and h is not title]
        columns.append(headers[-1].text if headers else "")
    table.columns = columns
    return table


def build_tables(document: Document) -> list[Table]:
    tables = []
    for page in document.pages:
        for index, group in enumerate(_segment(page.blocks), 1):
            tables.append(_build_table(page.page, index, group))
    document.tables = tables
    return tables


def find_table(document: Document, page: int, index: int) -> Table | None:
    return next((t for t in document.tables if t.page == page and t.index == index), None)
