from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Block:
    block_id: str
    page: int
    text: str
    bbox: tuple[float, float, float, float]
    table: int = 0
    row0: int = -1
    row1: int = -1
    col0: int = -1
    col1: int = -1

    @property
    def evidence(self) -> dict:
        return {"page": self.page, "bbox": list(self.bbox)}


@dataclass
class Page:
    page: int
    width: int
    height: int
    blocks: list[Block]
    image_path: Path | None = None


@dataclass
class Document:
    doc_id: str
    pages: list[Page]
    tables: list = field(default_factory=list)

    def page(self, number: int) -> Page | None:
        return next((p for p in self.pages if p.page == number), None)


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_documents(split_dir: Path) -> dict[str, Document]:
    documents = {}
    for manifest in read_jsonl(split_dir / "manifest.jsonl"):
        payload = json.loads((split_dir / manifest["ocr_path"]).read_text(encoding="utf-8"))
        images = {int(Path(p).stem.rsplit("_p", 1)[1]): split_dir / p for p in manifest["image_paths"]}
        pages = []
        for page in payload["pages"]:
            blocks = [Block(b["block_id"], int(b["page"]), str(b["text"]), tuple(float(v) for v in b["bbox"]))
                      for b in page["blocks"]]
            pages.append(Page(int(page["page"]), int(page["width"]), int(page["height"]), blocks,
                              images.get(int(page["page"]))))
        documents[manifest["id"]] = Document(manifest["id"], pages)
    return documents


def load_questions(split_dir: Path) -> list[dict]:
    return read_jsonl(split_dir / "questions.jsonl")


def load_labels(split_dir: Path) -> dict[str, dict]:
    path = split_dir / "labels.jsonl"
    return {row["question_id"]: row for row in read_jsonl(path)} if path.exists() else {}


def load_cell_annotations(split_dir: Path) -> dict[str, dict[str, dict]]:
    path = split_dir / "cell_annotations.jsonl"
    result: dict[str, dict[str, dict]] = {}
    if path.exists():
        for row in read_jsonl(path):
            result.setdefault(row["document_id"], {})[row["block_id"]] = row
    return result
