import math
import random

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForCausalLM.from_pretrained(cfg.model_name, torch_dtype=dtype).to(device).eval()
    start = tok.bos_token_id if tok.bos_token_id is not None else tok.eos_token_id
    print(f"model {cfg.model_name} on {device} {dtype}, start token {start}")
    return tok, model, device, start


def shuffles(words, n, rng):
    out = []
    for _ in range(n):
        perm = words[:]
        for _ in range(10):
            rng.shuffle(perm)
            if perm != words:
                break
        out.append(" ".join(perm))
    return out


@torch.inference_mode()
def log_likelihoods(texts, tok, model, device, start, batch_size):
    ids = [[start] + tok(t, add_special_tokens=False)["input_ids"] for t in texts]
    totals, counts = np.zeros(len(texts)), np.zeros(len(texts))
    order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
    for b in range(0, len(order), batch_size):
        idx = order[b:b + batch_size]
        width = max(len(ids[i]) for i in idx)
        x = torch.full((len(idx), width), start, dtype=torch.long)
        mask = torch.zeros((len(idx), width), dtype=torch.long)
        for row, i in enumerate(idx):
            x[row, :len(ids[i])] = torch.tensor(ids[i])
            mask[row, :len(ids[i])] = 1
        x, mask = x.to(device), mask.to(device)
        logits = model(input_ids=x, attention_mask=mask).logits[:, :-1].float()
        logp = torch.log_softmax(logits, -1).gather(-1, x[:, 1:, None])[..., 0]
        keep = mask[:, 1:].float()
        totals[idx] = (logp * keep).sum(1).cpu().numpy()
        counts[idx] = keep.sum(1).cpu().numpy()
    return totals, counts


def score_texts(cfg, texts, tok, model, device, start):
    rng = random.Random(cfg.seed)
    batch = []
    for t in texts:
        batch.append(t)
        batch.extend(shuffles(t.split(" "), cfg.n_shuffles, rng))
    totals, counts = log_likelihoods(batch, tok, model, device, start, cfg.batch_size)
    k = cfg.n_shuffles + 1
    rows = []
    for j, t in enumerate(texts):
        ll, n = totals[j * k], counts[j * k]
        others = totals[j * k + 1:(j + 1) * k]
        sd = float(others.std()) or 1.0
        words = t.split(" ")
        rows.append({"ll": float(ll), "ll_per_token": float(ll / n), "z": float((ll - others.mean()) / sd),
                     "rank_frac": float((others >= ll).mean()),
                     "start_word": float(words[0] in cfg.start_words),
                     "end_function_word": float(words[-1] in cfg.function_words)})
        assert math.isfinite(rows[-1]["z"]), t
    return rows
