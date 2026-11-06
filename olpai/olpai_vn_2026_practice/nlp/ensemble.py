from __future__ import annotations

import csv
import hashlib
from dataclasses import replace
from pathlib import Path

import torch

from nlp.config import Config, find_data_root
from nlp.data import clean, split_pairs
from nlp.engine import evaluate, translate_ids
from nlp.main import finalize, load_data
from nlp.model import build_model, check_incremental
from nlp.tokenizer import decode, encode, load_spm


def load_run(folder: Path, vi_size: int, device):
    ckpt = torch.load(folder / "model.pt", map_location="cpu", weights_only=False)
    cfg = Config.from_saved(ckpt["config"])
    sp_zh = load_spm(folder / "spm_zh.model")
    model = build_model(cfg, sp_zh.get_piece_size(), vi_size)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), sp_zh, cfg


def group_weights(zh_tokenizers: list[str], share: float | None) -> list[float]:
    primary = [t != "char" for t in zh_tokenizers]
    n_primary, n = sum(primary), len(primary)
    if share is None or n_primary in (0, n):
        return [1 / n] * n
    return [share / n_primary if p else (1 - share) / (n - n_primary) for p in primary]


def ensemble_runs(base: Config, run_dirs: dict[str, Path], output_dir: Path, log) -> dict:
    cfg = base.debug_variant() if base.debug else base
    cfg = replace(cfg, data_root=str(find_data_root(base.data_root)))
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vi_files = [d / "spm_vi.model" for d in run_dirs.values()]
    if len({hashlib.md5(p.read_bytes()).hexdigest() for p in vi_files}) > 1:
        raise ValueError(f"Runs use different vi tokenizers, cannot ensemble: {list(run_dirs)}")
    sp_vi = load_spm(vi_files[0])
    loaded = {n: load_run(d, sp_vi.get_piece_size(), device) for n, d in run_dirs.items()}
    models = [m for m, _, _ in loaded.values()]
    check_incremental(models[0])
    zh_kinds = [c.zh_tokenizer for _, _, c in loaded.values()]
    log(f"[ensemble] {len(models)} model: " + ", ".join(f"{n} (zh {k})" for n, k in zip(loaded, zh_kinds)))

    pairs, sources = load_data(cfg)
    _, val = split_pairs(pairs, cfg.val_size, cfg.dedupe_pairs, cfg.seed, log)
    val_zh, val_refs = [zh for zh, _ in val], [vi for _, vi in val]
    val_src = [encode(sp, val_zh, cfg.max_tokens) for _, sp, _ in loaded.values()]

    results, texts = {}, {}

    def try_config(beam: int, lp: float, share: float | None) -> float:
        weights = group_weights(zh_kinds, share)
        stats, hyps = evaluate(models, val_src, val_refs, sp_vi, cfg, device, beam, lp, weights)
        results[(beam, lp, share)], texts[(beam, lp, share)] = stats, hyps
        log(f"[ensemble] beam={beam} len_penalty={lp} weights={[round(w, 3) for w in weights]}: "
            f"BLEU val {stats['bleu']:.2f} | hyp/ref length {stats['ratio']:.3f} | BP {stats['bp']:.3f}")
        return stats["bleu"]

    first_beam = cfg.beam_grid[0]
    lp_scores = {lp: try_config(first_beam, lp, None) for lp in cfg.lp_grid}
    best_lp = max(lp_scores, key=lp_scores.get)
    share_scores = {None: lp_scores[best_lp]}
    for share in cfg.weight_grid:
        if len(set(group_weights(zh_kinds, share))) > 1:
            share_scores[share] = try_config(first_beam, best_lp, share)
    best_share = max(share_scores, key=share_scores.get)
    for beam in cfg.beam_grid[1:]:
        try_config(beam, best_lp, best_share)
    beam, lp, share = max(results, key=lambda k: results[k]["bleu"])
    weights = group_weights(zh_kinds, share)
    bleu = results[(beam, lp, share)]["bleu"]
    log(f"[ensemble] chose beam={beam} len_penalty={lp} weights={dict(zip(loaded, (round(w, 3) for w in weights)))}: "
        f"BLEU val {bleu:.2f}")

    with open(output_dir / "val_hyps.tsv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["zh", "ref", "hyp"])
        writer.writerows(zip(val_zh, val_refs, texts[(beam, lp, share)]))

    src_ids = [encode(sp, [clean(s) for s in sources], cfg.max_tokens) for _, sp, _ in loaded.values()]
    hyps = decode(sp_vi, translate_ids(models, src_ids, cfg, device, beam, lp, weights))
    hits, csv_path, zip_path = finalize(cfg, sources, hyps, pairs, output_dir, log)
    return {"bleu": bleu, "beam": beam, "len_penalty": lp, "weights": dict(zip(loaded, weights)),
            "grid": [{"beam": b, "len_penalty": l, "share": w, **v} for (b, l, w), v in results.items()],
            "runs": list(run_dirs), "memory_hits": hits,
            "submission_csv": str(csv_path), "submission_zip": str(zip_path), "output_dir": str(output_dir)}
