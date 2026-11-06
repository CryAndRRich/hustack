from __future__ import annotations

import re
from dataclasses import dataclass, field

Condition = tuple[str, str]

T = r"bảng (?P<table>\d+) ở trang (?P<page>\d+)"
CONDS = r"(?P<conds>.+”)"
PATTERNS: list[tuple[str, str, re.Pattern]] = [(name, kind, re.compile(p)) for name, kind, p in [
    ("lookup_1", "lookup", rf"Hãy cho biết (?P<target>.+?) tại {T} đối với {CONDS}\."),
    ("lookup_2", "lookup", rf"Trong {T}, (?P<target>.+?) của dòng có {CONDS} là gì\?"),
    ("lookup_3", "lookup", rf"Tại {T}, dòng {CONDS} ghi (?P<target>.+?) bằng bao nhiêu\?"),
    ("sum", "sum", rf"Trong {T}, tổng (?P<target>.+?) của hai dòng có {CONDS} là bao nhiêu\?"),
    ("argmax", "argmax", rf"Trong {T}, (?P<label>.+?) nào có (?P<target>.+) lớn nhất\?"),
    ("argmin", "argmin", rf"Trong {T}, (?P<label>.+?) nào có (?P<target>.+) nhỏ nhất\?"),
    ("compare", "compare",
     rf"So sánh tại {T}: giữa dòng có (?P<a>.+?”) với dòng có (?P<b>.+”), dòng nào có (?P<target>.+) cao hơn\?"),
    ("count", "count", rf"Có bao nhiêu dòng trong {T} có (?P<column>.+?) là “(?P<value>[^”]*)”\?"),
    ("cross_page_sum", "cross_page_sum",
     r"Lấy (?P<target>.+?) của dòng có (?P<a>.+?”) ở trang (?P<page_a>\d+) cộng với (?P<target_b>.+?) "
     r"của dòng có (?P<b>.+?”) ở trang (?P<page_b>\d+)\. Kết quả là bao nhiêu\?"),
    ("bold_1", "visual_bold_lookup",
     rf"Tại {T}, hai dòng {CONDS} có kiểu chữ khác nhau\. Hãy tìm dòng in đậm trên ảnh và trả về nội dung cột "
     r"(?P<target>.+)\."),
    ("bold_2", "visual_bold_lookup",
     rf"Chỉ dựa vào hình thức trình bày trên ảnh ở {T}: xác định dòng có toàn bộ chữ in đậm trong hai dòng "
     rf"{CONDS}; giá trị (?P<target>.+) của dòng đó là gì\?"),
    ("bold_3", "visual_bold_lookup",
     rf"OCR không ghi nhận định dạng chữ\. Trong ảnh của {T}, hãy chọn dòng in đậm giữa {CONDS}, "
     r"rồi đọc ô thuộc cột (?P<target>.+)\."),
    ("bold_4", "visual_bold_lookup",
     rf"Quan sát trực tiếp kiểu chữ trong ảnh tại {T}\. Giữa dòng (?P<a>.+?”) và dòng (?P<b>.+”), "
     r"dòng nào được in đậm\? Trả lời bằng giá trị ở cột (?P<target>.+)\."),
]]


@dataclass
class Query:
    question_id: str
    document_id: str
    text: str
    kind: str
    template: str = ""
    page: int = 0
    table: int = 0
    target: str = ""
    label: str = ""
    conditions: list[Condition] = field(default_factory=list)
    rows: list[list[Condition]] = field(default_factory=list)
    pages: list[int] = field(default_factory=list)
    targets: list[str] = field(default_factory=list)


def parse_conditions(text: str) -> list[Condition]:
    pairs: list[Condition] = []
    previous = 0
    for match in re.finditer(r"“([^”]*)”", text):
        header = text[previous:match.start()]
        header = re.sub(r"^\s*(?:và|với)\s+", "", header).strip(" ,;:")
        pairs.append((header, match.group(1)))
        previous = match.end()
    return pairs


def parse_question(item: dict) -> Query:
    text = item["question"].strip()
    query = Query(item["question_id"], item["document_id"], text, "unknown")
    for name, kind, pattern in PATTERNS:
        match = pattern.fullmatch(text)
        if not match:
            continue
        g = match.groupdict()
        query.kind, query.template = kind, name
        query.page = int(g.get("page") or 0)
        query.table = int(g.get("table") or 0)
        query.target = g.get("target") or ""
        query.label = g.get("label") or ""
        if kind == "count":
            query.conditions = [(g["column"], g["value"])]
        elif "a" in g and g.get("a"):
            query.rows = [parse_conditions(g["a"]), parse_conditions(g["b"])]
            query.conditions = query.rows[0] + query.rows[1]
        elif g.get("conds"):
            query.conditions = parse_conditions(g["conds"])
        if kind == "cross_page_sum":
            query.pages = [int(g["page_a"]), int(g["page_b"])]
            query.targets = [g["target"], g["target_b"]]
        return query
    return query
