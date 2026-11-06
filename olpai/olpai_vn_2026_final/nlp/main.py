from __future__ import annotations

import json
import tempfile
import zipfile
from dataclasses import asdict, replace
from pathlib import Path

import torch

from common.distributed import RESULT_FILE, cli, run_or_launch, run_ranks
from common.paths import unpack_split
from common.utils import Logger, seed_everything
from nlp.config import DOMAINS, Config, find_data_root, resolve_output_dir, setup_hf_offline
from nlp.data import Example, category_sets, load_examples, split_path, test_split_name, to_submission
from nlp.engine import fit, parse_all, predict_texts
from nlp.linearize import CategoryMapper
from nlp.metric import evaluate, format_scores, validate_submission
from nlp.model import encode, load_model, save_model

DEV_PREDICTIONS = "dev_predictions.jsonl"


def prepare(cfg: Config) -> Config:
    data_root = find_data_root(cfg.data_root)
    output_dir = resolve_output_dir(cfg.output_dir)
    cfg, output_dir = cfg.for_run(output_dir)
    test_file = cfg.test_file
    if not test_file and cfg.split == "private_test":
        root = unpack_split(data_root, cfg.split, cfg.zip_password, Path(tempfile.gettempdir()) / "tacvu1_unpacked")
        test_file = str(split_path(root, test_split_name(cfg.split)))
    return replace(cfg, data_root=str(data_root), output_dir=str(output_dir), test_file=test_file)


def run(cfg: Config | None = None) -> dict:
    cfg = prepare(cfg or Config())
    setup_hf_offline()
    output_dir = Path(cfg.output_dir)
    log = Logger(output_dir / "train.log")
    return run_or_launch(cfg, "nlp.main", execute, log)


def execute(cfg: Config) -> dict | None:
    setup_hf_offline()
    return run_ranks(_execute, cfg)


def load_data(cfg: Config, log):
    root = Path(cfg.data_root)
    train = load_examples(split_path(root, "Train"))
    dev = load_examples(split_path(root, "Dev"))
    test_path = Path(cfg.test_file) if cfg.test_file else split_path(root, test_split_name(cfg.split))
    test = load_examples(test_path)
    if cfg.debug:
        train, dev, test = ([ex for ex in exs if ex.index < n] for exs, n in ((train, 32), (dev, 16), (test, 8)))
        log("[debug] running on a subset — RESULTS NOT FOR SUBMISSION")
    count = lambda exs: {d: sum(ex.domain == d for ex in exs) for d in DOMAINS}
    log(f"[data] train {len(train)} {count(train)} | dev {len(dev)} | {cfg.split} {len(test)} ({test_path})")
    return train, dev, test


def _execute(cfg: Config, ctx) -> dict | None:
    output_dir = Path(cfg.output_dir)
    log = Logger(output_dir / "train.log", enabled=ctx.is_main)
    seed_everything(cfg.seed)
    device = ctx.device
    gpu = f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""
    log(f"[config] {json.dumps(asdict(cfg), ensure_ascii=False)}")
    log(f"[config] device={device}{gpu} × {ctx.world_size} | torch {torch.__version__}")

    train, dev, test = load_data(cfg, log)
    mapper = CategoryMapper(category_sets(train, dev))

    if not ctx.is_main:
        ctx.barrier()
    tokenizer, model = load_model(cfg, cfg.checkpoint, log)
    if ctx.is_main:
        ctx.barrier()
    model.to(device)
    torch.manual_seed(cfg.seed)

    dev_items = encode(cfg, tokenizer, dev, with_target=True)
    test_items = encode(cfg, tokenizer, test, with_target=False)
    if cfg.checkpoint:
        best = {"epoch": None}
        log(f"[checkpoint] {cfg.checkpoint}: skipping training")
    else:
        fit_examples = train + dev if cfg.train_on_dev else train
        train_items = encode(cfg, tokenizer, fit_examples, with_target=True)
        src_len = sorted(len(it.source) for it in train_items)
        tgt_len = sorted(len(it.target) for it in train_items)
        log(f"[tokenize] source: mean {sum(src_len) / len(src_len):.0f}, max {src_len[-1]} token | target: "
            f"mean {sum(tgt_len) / len(tgt_len):.0f}, max {tgt_len[-1]} (limit {cfg.max_target_len})"
            + (" | train includes Dev" if cfg.train_on_dev else ""))
        best = fit(cfg, ctx, model, tokenizer, train_items, dev, dev_items, mapper, log)

    val = {"best_epoch": best["epoch"]}
    dev_texts = None
    if not cfg.train_on_dev:
        dev_texts = predict_texts(model, tokenizer, dev_items, cfg, ctx, cfg.num_beams)
        scores = evaluate(dev, parse_all(dev_texts, dev, mapper, cfg))
        val.update(avg_f1=scores["avg_f1"], per_domain={d: round(scores[d]["f1"], 4) for d in DOMAINS})
        log(f"[dev] epoch {best['epoch']}, beam {cfg.num_beams}:\n" + format_scores(scores))
    test_texts = predict_texts(model, tokenizer, test_items, cfg, ctx, cfg.num_beams)

    result = None
    if ctx.is_main:
        if dev_texts is not None:
            write_predictions(output_dir / DEV_PREDICTIONS, dev, dev_texts, parse_all(dev_texts, dev, mapper, cfg))
        test_preds = parse_all(test_texts, test, mapper, cfg)
        write_predictions(output_dir / f"{cfg.split}_predictions.jsonl", test, test_texts, test_preds)
        json_path, zip_path = write_submission(test, test_preds, output_dir, log)
        empty = sum(1 for ex in test if not test_preds[(ex.domain, ex.index)])
        log(f"[{cfg.split}] {sum(map(len, test_preds.values()))} triples / {len(test)} sentences | sentences without triples: {empty}")
        result = {"val": val, "submission_json": str(json_path), "submission_zip": str(zip_path),
                  "output_dir": str(output_dir), "config": asdict(cfg)}
    if cfg.save_model and not cfg.checkpoint and ctx.is_main:
        save_model(model, tokenizer, output_dir / "model")
        log(f"[save] model fp16 -> {output_dir / 'model'} (reuse via checkpoint=...)")
    ctx.barrier()
    if result is not None:
        (output_dir / RESULT_FILE).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def write_predictions(path: Path, examples: list[Example], texts: list[str], preds: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for ex, text in zip(examples, texts):
            row = {"domain": ex.domain, "index": ex.index, "raw_words": ex.raw_words, "generated": text,
                   "triples": preds[(ex.domain, ex.index)], "gold": ex.triples}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_predictions(path: Path) -> dict[tuple[str, int], list[tuple[str, str, str]]]:
    preds = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            preds[(row["domain"], row["index"])] = [tuple(t) for t in row["triples"]]
    return preds


def write_submission(test: list[Example], preds: dict, output_dir: Path, log=print):
    submission = to_submission(test, preds)
    validate_submission(submission, test)
    json_path, zip_path = output_dir / "submission.json", output_dir / "submission.zip"
    json_path.write_text(json.dumps(submission, ensure_ascii=False, indent=2), encoding="utf-8")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(json_path, arcname="submission.json")
    log(f"[submit] {zip_path} | " + ", ".join(f"{d}: {len(submission[d])}" for d in DOMAINS))
    return json_path, zip_path


if __name__ == "__main__":
    cli(Config, run, execute)
