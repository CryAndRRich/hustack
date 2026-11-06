from __future__ import annotations

SUBMISSION_COLUMNS = ["tieng_trung", "tieng_viet"]


def validate_submission(rows: list[tuple[str, str]], sources: list[str]) -> None:
    if len(rows) != len(sources):
        raise ValueError(f"Row count {len(rows)} != source sentence count {len(sources)}")
    bad = [i for i, ((src, _), ref) in enumerate(zip(rows, sources)) if src != ref]
    if bad:
        raise ValueError(f"Column tieng_trung differs from the source in {len(bad)} rows (e.g. row {bad[0]})")
    if any("\n" in vi or "\r" in vi for _, vi in rows):
        raise ValueError("Translation contains a newline")


def bleu_stats(hyps: list[str], refs: list[str]) -> dict:
    import sacrebleu

    result = sacrebleu.corpus_bleu(hyps, [refs], force=True)
    return {"bleu": float(result.score), "ratio": result.sys_len / max(1, result.ref_len), "bp": float(result.bp)}
