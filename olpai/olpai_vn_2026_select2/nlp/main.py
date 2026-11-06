from __future__ import annotations

import json
import random
import shutil
import sys
import time
import zipfile
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from common.distributed import RESULT_FILE
from common.paths import PROJECT_ROOT
from common.utils import Logger
from nlp.bold import CellScoreScorer, StrokeScorer
from nlp.config import Config, find_data_root, resolve_output_dir
from nlp.data import Document, load_cell_annotations, load_documents, load_labels, load_questions
from nlp.metric import evaluate, validate_predictions
from nlp.question import Query, parse_question
from nlp.solver import solve
from nlp.table import build_tables, find_table


def prepare(cfg: Config) -> Config:
    data_root = find_data_root(cfg.data_root)
    output_dir = resolve_output_dir(cfg.output_dir)
    cfg, output_dir = cfg.for_run(output_dir)
    cache_dir = cfg.cache_dir_or("tacvu2_cache")
    return replace(cfg, data_root=str(data_root), output_dir=str(output_dir), cache_dir=str(cache_dir))


def load_split(split_dir: Path) -> tuple[dict[str, Document], list[Query]]:
    if not (split_dir / "manifest.jsonl").exists():
        hint = " (private_test.zip not extracted?)" if split_dir.with_suffix(".zip").exists() else ""
        raise FileNotFoundError(f"{split_dir / 'manifest.jsonl'} not found{hint}")
    docs = load_documents(split_dir)
    for doc in docs.values():
        build_tables(doc)
    return docs, [parse_question(q) for q in load_questions(split_dir)]


def split_documents(doc_ids: list[str], fraction: float, seed: int) -> tuple[set[str], set[str]]:
    ids = sorted(doc_ids)
    random.Random(seed).shuffle(ids)
    n_val = max(1, round(len(ids) * fraction))
    return set(ids[n_val:]), set(ids[:n_val])


def data_cells(doc: Document, page: int | None = None):
    for table in doc.tables:
        if page is None or table.page == page:
            yield from (b for b in table.blocks if b.row0 >= table.header_rows)


def _bold_scorer_cnn(cfg: Config, train_docs, ann, train_ids, val_ids, eval_items, log):
    import torch

    from common.distributed import launch_torchrun
    from nlp import bold

    cache_dir, out = Path(cfg.cache_dir), Path(cfg.output_dir)
    rng = random.Random(cfg.seed)

    def sample(doc_ids: set[str]):
        pages, labels = [], []
        for doc_id in sorted(doc_ids):
            doc = train_docs[doc_id]
            for page in doc.pages:
                cells = list(data_cells(doc, page.page))
                pos = [b for b in cells if ann[doc_id][b.block_id]["is_bold"]]
                neg = [b for b in cells if not ann[doc_id][b.block_id]["is_bold"]]
                k = min(len(neg), max(4, round(len(pos) * cfg.neg_per_pos)))
                chosen = pos + rng.sample(neg, k)
                pages.append((page.image_path, [(f"{doc_id}|{b.block_id}", b.bbox) for b in chosen]))
                labels += [1] * len(pos) + [0] * k
        return pages, np.array(labels, dtype=np.uint8)

    size = dict(height=cfg.crop_height, width=cfg.crop_width, workers=cfg.decode_workers, log=log)
    job = {"config": asdict(cfg), "scores_out": str(out / "bold_scores.npy")}
    if not cfg.checkpoint:
        for name, ids in (("train", train_ids), ("val", val_ids)):
            pages, labels = sample(ids)
            path, _ = bold.build_crop_cache(pages, cache_dir, f"bold_{name}", **size)
            label_path = path.with_name(path.stem + "_labels.npy")
            np.save(label_path, labels)
            job[f"{name}_cache"], job[f"{name}_labels"] = str(path), str(label_path)
    eval_path, eval_keys = bold.build_crop_cache(eval_items, cache_dir, "bold_eval", **size)
    job["eval_cache"] = str(eval_path)

    job_path = out / "bold_job.json"
    job_path.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
    n_gpu = torch.cuda.device_count()
    if cfg.ddp and n_gpu > 1:
        log(f"[ddp] launching torchrun on {n_gpu} GPUs for the bold CNN")
        launch_torchrun(n_gpu, job_path, cwd=PROJECT_ROOT, module="nlp.bold")
    else:
        bold.execute(job)
    logits = np.load(job["scores_out"])
    return CellScoreScorer(dict(zip(eval_keys, logits.tolist())))


def _eval_items(queries: list[Query], docs: dict[str, Document]):
    pages: dict[tuple[str, int], dict[str, tuple]] = {}
    for q in queries:
        if q.kind != "visual_bold_lookup":
            continue
        doc = docs[q.document_id]
        table = find_table(doc, q.page, q.table)
        if table is None:
            continue
        cells = pages.setdefault((doc.doc_id, table.page), {})
        for b in table.blocks:
            if b.row0 >= table.header_rows:
                cells[f"{doc.doc_id}|{b.block_id}"] = b.bbox
    return [(docs[d].page(p).image_path, sorted(cells.items())) for (d, p), cells in sorted(pages.items())]


def _format_eval(result: dict) -> str:
    order = ["ALL", "lookup", "argmax", "argmin", "sum", "compare", "count", "visual_bold_lookup", "cross_page_sum"]
    lines = [f"{'kind':20s} {'n':>5s} {'score':>7s} {'anls':>7s} {'evid':>7s}"]
    for kind in [k for k in order if k in result]:
        r = result[kind]
        lines.append(f"{kind:20s} {r['n']:5d} {r['score']:7.2f} {r['anls']:7.2f} {r['evidence_f1']:7.2f}")
    return "\n".join(lines)


def run(cfg: Config | None = None) -> dict:
    cfg = prepare(cfg or Config())
    out = Path(cfg.output_dir)
    log = Logger(out / "train.log")
    log(f"[config] {json.dumps(asdict(cfg), ensure_ascii=False)}")
    started = time.time()

    data_root = Path(cfg.data_root)
    train_docs, train_q = load_split(data_root / "training_set")
    labels = load_labels(data_root / "training_set")
    target_docs, target_q = load_split(data_root / cfg.split)
    train_ids, val_ids = split_documents(list(train_docs), cfg.val_docs, cfg.seed)
    if cfg.debug:
        train_ids, val_ids = set(sorted(train_ids)[:60]), set(sorted(val_ids)[:30])
        keep = set(sorted(target_docs)[:20])
        target_q = [q for q in target_q if q.document_id in keep]
        log("[debug] running on a subset — RESULTS NOT FOR SUBMISSION")
    val_q = [q for q in train_q if q.document_id in val_ids]
    log(f"[data] train {len(train_ids)} docs | val {len(val_ids)} docs ({len(val_q)} questions) | "
        f"{cfg.split}: {len(target_docs)} docs, {len(target_q)} questions | "
        f"question kinds {cfg.split}: {dict(Counter(q.kind for q in target_q))} | {time.time() - started:.0f}s")

    if cfg.bold_mode == "cnn":
        ann = load_cell_annotations(data_root / "training_set")
        all_docs = {**train_docs, **target_docs}
        scorer = _bold_scorer_cnn(cfg, train_docs, ann, train_ids, val_ids,
                                  _eval_items(val_q + target_q, all_docs), log)
    else:
        scorer = StrokeScorer()

    val_preds = {q.question_id: solve(train_docs[q.document_id], q, scorer).to_json(q.question_id) for q in val_q}
    val_result = evaluate(val_preds, {q.question_id: labels[q.question_id] for q in val_q})
    log("[val] score by question kind (out of 100):\n" + _format_eval(val_result))
    bold_used = cfg.bold_mode
    bold_q = [q for q in val_q if q.kind == "visual_bold_lookup"]
    if cfg.bold_mode == "cnn" and bold_q:
        stroke = StrokeScorer()
        stroke_preds = {q.question_id: solve(train_docs[q.document_id], q, stroke).to_json(q.question_id) for q in bold_q}
        stroke_anls = evaluate(stroke_preds, {q.question_id: labels[q.question_id] for q in bold_q})["ALL"]["anls"]
        cnn_anls = val_result["visual_bold_lookup"]["anls"]
        if cfg.bold_fallback and stroke_anls > cnn_anls:
            scorer, bold_used = stroke, "stroke"
        log(f"[val] visual_bold_lookup ANLS: CNN {cnn_anls:.2f} | heuristic stroke {stroke_anls:.2f} "
            f"-> using {bold_used} for {cfg.split}")

    answers = [(q, solve(target_docs[q.document_id], q, scorer)) for q in target_q]
    rows = [a.to_json(q.question_id) for q, a in answers]
    unsolved = Counter(q.kind for q, a in answers if not a.solved)
    log(f"[predict] {cfg.split}: {len(rows)} questions | unsolved: {sum(unsolved.values())} {dict(unsolved)}")
    expected = [q.question_id for q in target_q]
    validate_predictions(rows, expected, {d: len(doc.pages) for d, doc in target_docs.items()})

    pred_path, zip_path = out / "predictions.jsonl", out / "submission.zip"
    with open(pred_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(pred_path, arcname="predictions.jsonl")
    if cfg.split == "private_test":
        shutil.copyfile(zip_path, out / "private_submission.zip")
    log(f"[submit] {zip_path} | total time {time.time() - started:.0f}s")

    result = {"val": val_result, "bold_used": bold_used, "unsolved": dict(unsolved), "predictions": str(pred_path),
              "submission_zip": str(zip_path), "output_dir": str(out), "config": asdict(cfg)}
    (out / RESULT_FILE).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    run(Config().with_overrides(sys.argv[1:]))
