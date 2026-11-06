from __future__ import annotations

import csv
import json
import shutil
import sys
import zipfile
from dataclasses import asdict, replace
from pathlib import Path

import torch

from common.distributed import RESULT_FILE
from common.utils import Logger, seed_everything
from nlp.config import Config, find_data_root, resolve_output_dir
from nlp.data import build_memory, clean, load_pairs, load_test, split_pairs
from nlp.engine import average_states, evaluate, fit, translate_ids
from nlp.metric import validate_submission
from nlp.model import build_model, check_incremental
from nlp.tokenizer import SAMPLING_TYPES, decode, encode, load_spm, train_spm

SUBMISSION_CSV = "nlp_submission.csv"
SUBMISSION_ZIP = "nlp_submission.zip"


def prepare(cfg: Config) -> Config:
    data_root = find_data_root(cfg.data_root)
    output_dir = resolve_output_dir(cfg.output_dir)
    if cfg.debug:
        cfg = cfg.debug_variant()
        if not cfg.output_dir:
            output_dir = output_dir / "debug"
    output_dir.mkdir(parents=True, exist_ok=True)
    return replace(cfg, data_root=str(data_root), output_dir=str(output_dir))


def load_data(cfg: Config) -> tuple[list[tuple[str, str]], list[str]]:
    data_root = Path(cfg.data_root)
    pairs, sources = load_pairs(data_root), load_test(data_root, cfg.split)
    return (pairs[:3000], sources[:20]) if cfg.debug else (pairs, sources)


def finalize(cfg: Config, sources: list[str], hyps: list[str], pairs, output_dir: Path, log) -> tuple[int, Path, Path]:
    memory = build_memory(pairs) if cfg.use_memory else {}
    hits = 0
    for i, src in enumerate(map(clean, sources)):
        if src in memory:
            hyps[i] = memory[src]
            hits += 1
    empty = sum(not h for h in hyps)
    log(f"[translate] {cfg.split}: {len(sources)} sentences | translation memory replaced {hits}"
        + (f" | {empty} empty translations" if empty else ""))
    (output_dir / f"{cfg.split}_hyps.txt").write_text("\n".join(hyps) + "\n", encoding="utf-8")
    csv_path, zip_path = write_submission(sources, hyps, output_dir, log)
    return hits, csv_path, zip_path


def run(cfg: Config | None = None) -> dict:
    cfg = prepare(cfg or Config())
    output_dir = Path(cfg.output_dir)
    log = Logger(output_dir / "train.log")
    seed_everything(cfg.seed + cfg.run_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu = f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""
    log(f"[config] {json.dumps(asdict(cfg), ensure_ascii=False)}")
    log(f"[config] device={device}{gpu} | torch {torch.__version__}")

    pairs, sources = load_data(cfg)
    if cfg.debug:
        log("[debug] subset — RESULTS NOT FOR SUBMISSION")

    if cfg.checkpoint:
        model, sp_zh, sp_vi, val = _load_checkpoint(cfg, device, log)
    else:
        model, sp_zh, sp_vi, val = _train(cfg, pairs, device, log)

    src_ids = encode(sp_zh, [clean(s) for s in sources], cfg.max_tokens)
    hyps = decode(sp_vi, translate_ids([model], [src_ids], cfg, device, cfg.beam))
    hits, csv_path, zip_path = finalize(cfg, sources, hyps, pairs, output_dir, log)

    result = {"val": val, "memory_hits": hits, "submission_csv": str(csv_path), "submission_zip": str(zip_path),
              "output_dir": str(output_dir), "config": asdict(cfg)}
    (output_dir / RESULT_FILE).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _train(cfg: Config, pairs, device, log):
    output_dir = Path(cfg.output_dir)
    train, val = split_pairs(pairs, cfg.val_size, cfg.dedupe_pairs, cfg.seed, log)

    sp_zh = load_spm(train_spm([zh for zh, _ in train], output_dir / "spm_zh", cfg.zh_tokenizer, cfg.zh_vocab,
                               byte_fallback=True))
    if cfg.vi_spm:
        shutil.copy(cfg.vi_spm, output_dir / "spm_vi.model")
        sp_vi = load_spm(output_dir / "spm_vi.model")
    else:
        sp_vi = load_spm(train_spm([vi for _, vi in train], output_dir / "spm_vi", cfg.vi_tokenizer, cfg.vi_vocab,
                                   byte_fallback=False))
    zh_texts, vi_texts = [zh for zh, _ in train], [vi for _, vi in train]
    train_src = encode(sp_zh, zh_texts, cfg.max_tokens)
    train_tgt = encode(sp_vi, vi_texts, cfg.max_tokens)
    val_src = encode(sp_zh, [zh for zh, _ in val], cfg.max_tokens)
    val_refs = [vi for _, vi in val]
    roundtrip = sum(a != b for a, b in zip(decode(sp_vi, train_tgt[:2000]), [vi for _, vi in train[:2000]]))
    log(f"[tokenizer] zh {cfg.zh_tokenizer} {sp_zh.get_piece_size()} | vi {cfg.vi_tokenizer} {sp_vi.get_piece_size()} "
        f"| tokens/sentence zh {sum(map(len, train_src)) / len(train_src):.1f}, vi {sum(map(len, train_tgt)) / len(train_tgt):.1f} "
        f"| vi decode not verbatim {roundtrip}/2000 sentences")

    zh_drop = cfg.subword_dropout if cfg.zh_tokenizer in SAMPLING_TYPES else 0.0
    vi_drop = cfg.subword_dropout if cfg.vi_tokenizer in SAMPLING_TYPES else 0.0
    resample = None
    if zh_drop or vi_drop:
        def resample(epoch):
            return encode(sp_zh, zh_texts, cfg.max_tokens, zh_drop), encode(sp_vi, vi_texts, cfg.max_tokens, vi_drop)
        log(f"[tokenizer] BPE-dropout per epoch: zh {zh_drop}, vi {vi_drop}")

    model = build_model(cfg, sp_zh.get_piece_size(), sp_vi.get_piece_size()).to(device)
    check_incremental(model)
    trained = fit(cfg, model, train_src, train_tgt, val_src, val_refs, sp_vi, device, log, resample=resample)

    candidates = {"best": trained["best"][2]}
    if len(trained["top"]) > 1:
        candidates["avg"] = average_states([s for _, _, s in trained["top"]])
    scores = {}
    for name, state in candidates.items():
        model.load_state_dict(state)
        stats, hyps = evaluate([model], [val_src], val_refs, sp_vi, cfg, device, beam=cfg.beam)
        scores[name] = stats["bleu"]
        if name == max(scores, key=scores.get):
            best_hyps = hyps
    chosen = max(scores, key=scores.get)
    model.load_state_dict(candidates[chosen])
    log(f"[val] BLEU beam {cfg.beam}: " + " | ".join(f"{k}={v:.2f}" for k, v in scores.items()) + f" -> chose {chosen}")

    with open(output_dir / "val_hyps.tsv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["zh", "ref", "hyp"])
        writer.writerows((zh, ref, hyp) for (zh, ref), hyp in zip(val, best_hyps))
    torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()}, "config": asdict(cfg)},
               output_dir / "model.pt")
    val = {"bleu": scores[chosen], "chosen": chosen, "bleu_all": scores,
           "greedy_top": [(round(b, 2), e) for b, e, _ in trained["top"]]}
    return model, sp_zh, sp_vi, val


def _load_checkpoint(cfg: Config, device, log):
    folder = Path(cfg.checkpoint)
    ckpt = torch.load(folder / "model.pt", map_location="cpu", weights_only=False)
    saved = Config.from_saved(ckpt["config"], data_root=cfg.data_root, output_dir=cfg.output_dir,
                              checkpoint=cfg.checkpoint, split=cfg.split, debug=cfg.debug)
    sp_zh, sp_vi = load_spm(folder / "spm_zh.model"), load_spm(folder / "spm_vi.model")
    model = build_model(saved, sp_zh.get_piece_size(), sp_vi.get_piece_size())
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    check_incremental(model)
    log(f"[checkpoint] {folder}")
    return model, sp_zh, sp_vi, {}


def write_submission(sources: list[str], hyps: list[str], output_dir: Path, log=print):
    rows = list(zip(sources, hyps))
    validate_submission(rows, sources)
    csv_path, zip_path = output_dir / SUBMISSION_CSV, output_dir / SUBMISSION_ZIP
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tieng_trung", "tieng_viet"])
        writer.writerows(rows)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(csv_path, arcname=SUBMISSION_CSV)
    log(f"[submit] {zip_path} | {len(rows)} sentences")
    return csv_path, zip_path


if __name__ == "__main__":
    args = sys.argv[1:]
    if args[:1] == ["--config"]:
        run(Config(**json.loads(Path(args[1]).read_text(encoding="utf-8"))))
    else:
        run(Config().with_overrides(args))
