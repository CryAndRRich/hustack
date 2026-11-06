from __future__ import annotations

from pathlib import Path

PAD, UNK, BOS, EOS = 0, 1, 2, 3


def train_spm(sentences: list[str], prefix: Path, model_type: str, vocab_size: int, byte_fallback: bool) -> Path:
    import sentencepiece as spm

    prefix = Path(prefix)
    corpus = prefix.with_suffix(".txt")
    corpus.write_text("\n".join(sentences) + "\n", encoding="utf-8")
    spm.SentencePieceTrainer.train(
        input=str(corpus), model_prefix=str(prefix), model_type=model_type, vocab_size=vocab_size,
        character_coverage=1.0, normalization_rule_name="identity", hard_vocab_limit=False,
        byte_fallback=byte_fallback and model_type in ("bpe", "unigram"),
        pad_id=PAD, unk_id=UNK, bos_id=BOS, eos_id=EOS, num_threads=4, minloglevel=2,
    )
    corpus.unlink()
    return prefix.with_suffix(".model")


def load_spm(path: Path):
    import sentencepiece as spm

    return spm.SentencePieceProcessor(model_file=str(path))


SAMPLING_TYPES = ("bpe", "unigram")


def encode(processor, sentences: list[str], max_tokens: int, dropout: float = 0.0) -> list[list[int]]:
    if dropout > 0:
        ids = processor.encode(sentences, enable_sampling=True, alpha=dropout, nbest_size=-1)
    else:
        ids = processor.encode(sentences)
    return [x[:max_tokens - 1] for x in ids]


def decode(processor, ids: list[list[int]]) -> list[str]:
    return [" ".join(text.split()) for text in processor.decode(ids)]
