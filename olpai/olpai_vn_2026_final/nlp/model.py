from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from nlp.data import Example
from nlp.linearize import source_text, target_text

IGNORE = -100


@dataclass
class Item:
    key: tuple[str, int]
    source: list[int]
    target: list[int]


def amp_mode(cfg, device: torch.device):
    if device.type != "cuda":
        return False, torch.float32
    major, _ = torch.cuda.get_device_capability(device)
    if major >= 8:
        return cfg.amp is not False, torch.bfloat16
    enabled = cfg.amp if cfg.amp is not None else not cfg.is_seq2seq
    return bool(enabled), torch.float16


def load_model(cfg, source: str | None = None, log=print):
    from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

    source = source or cfg.model_name
    tokenizer = AutoTokenizer.from_pretrained(source)
    model_cls = AutoModelForSeq2SeqLM if cfg.is_seq2seq else AutoModelForCausalLM
    model = model_cls.from_pretrained(source, torch_dtype=torch.float32)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    log(f"[model] {source} | {type(model).__name__} | {n_params:.0f}M params")
    return tokenizer, model


def save_model(model, tokenizer, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.to(torch.float16).save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)


def encode(cfg, tokenizer, examples: list[Example], with_target: bool) -> list[Item]:
    causal = not cfg.is_seq2seq
    items = []
    for ex in examples:
        if causal:
            prefix = tokenizer(f"Domain: {ex.domain}. Review: ", add_special_tokens=False)["input_ids"]
            review = tokenizer(ex.raw_words, add_special_tokens=False)["input_ids"]
            suffix = tokenizer("\nTriples:", add_special_tokens=False)["input_ids"]
            budget = max(8, cfg.max_source_len - len(prefix) - len(suffix))
            source = prefix + review[:budget] + suffix
        else:
            source = tokenizer(source_text(ex, causal=False), max_length=cfg.max_source_len,
                               truncation=True)["input_ids"]
        target = []
        if with_target:
            text = target_text(ex.triples, cfg.verbalize)
            if causal:
                ids = tokenizer(" " + text, add_special_tokens=False)["input_ids"]
                target = ids[:cfg.max_target_len - 1] + [tokenizer.eos_token_id]
            else:
                target = tokenizer(text_target=text, max_length=cfg.max_target_len, truncation=True)["input_ids"]
        items.append(Item((ex.domain, ex.index), source, target))
    return items


def _pad(seqs: list[list[int]], value: int, left: bool = False) -> torch.Tensor:
    width = max(len(s) for s in seqs)
    rows = [([value] * (width - len(s)) + s) if left else (s + [value] * (width - len(s))) for s in seqs]
    return torch.tensor(rows, dtype=torch.long)


class Collator:
    def __init__(self, pad_id: int, seq2seq: bool):
        self.pad_id = pad_id
        self.seq2seq = seq2seq

    def __call__(self, items: list[Item]) -> dict[str, torch.Tensor]:
        if self.seq2seq:
            input_ids = _pad([it.source for it in items], self.pad_id)
            labels = _pad([it.target for it in items], IGNORE)
        else:
            input_ids = _pad([it.source + it.target for it in items], self.pad_id)
            labels = _pad([[IGNORE] * len(it.source) + it.target for it in items], IGNORE)
        attention = _pad([[1] * (len(it.source) + (0 if self.seq2seq else len(it.target))) for it in items], 0)
        return {"input_ids": input_ids, "attention_mask": attention, "labels": labels}


@torch.inference_mode()
def generate(model, tokenizer, items: list[Item], cfg, device, num_beams: int, use_amp: bool, amp_dtype) -> list[str]:
    model.eval()
    seq2seq = cfg.is_seq2seq
    order = sorted(range(len(items)), key=lambda i: len(items[i].source))
    outputs = [""] * len(items)
    pad_id = tokenizer.pad_token_id
    for start in range(0, len(order), cfg.gen_batch_size):
        batch = [items[i] for i in order[start:start + cfg.gen_batch_size]]
        input_ids = _pad([it.source for it in batch], pad_id, left=not seq2seq).to(device)
        attention = _pad([[1] * len(it.source) for it in batch], 0, left=not seq2seq).to(device)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            generated = model.generate(
                input_ids=input_ids, attention_mask=attention, max_new_tokens=cfg.max_target_len,
                num_beams=num_beams, do_sample=False, early_stopping=num_beams > 1, use_cache=True,
                pad_token_id=pad_id, eos_token_id=tokenizer.eos_token_id,
            )
        if not seq2seq:
            generated = generated[:, input_ids.shape[1]:]
        texts = tokenizer.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        for i, text in zip(order[start:start + cfg.gen_batch_size], texts):
            outputs[i] = text.strip()
    return outputs
