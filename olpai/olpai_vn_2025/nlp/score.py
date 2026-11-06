from __future__ import annotations

import csv
import time
from itertools import combinations
from pathlib import Path

import torch


class BiEncoder:
    def __init__(self, name: str, path: str, prefix: str, device: torch.device, max_length: int, batch: int):
        from sentence_transformers import SentenceTransformer

        self.name, self.prefix, self.batch = name, prefix, batch
        self.model = SentenceTransformer(path, device=str(device))
        self.model.half()
        self.model.max_seq_length = max_length

    @torch.inference_mode()
    def encode(self, texts: list[str]) -> torch.Tensor:
        if not texts:
            return torch.empty(0, 1, device=self.model.device)
        emb = self.model.encode([self.prefix + t for t in texts], batch_size=self.batch, convert_to_tensor=True,
                                normalize_embeddings=True, show_progress_bar=False)
        return emb.float()


class Reranker:
    def __init__(self, path: str, device: torch.device, max_length: int, batch: int):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path, torch_dtype=torch.float16).to(device).eval()
        self.device, self.max_length, self.batch = device, max_length, batch

    @torch.inference_mode()
    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][0]) + len(pairs[i][1]))
        out = [0.0] * len(pairs)
        for start in range(0, len(order), self.batch):
            idx = order[start:start + self.batch]
            enc = self.tokenizer([pairs[i][0] for i in idx], [pairs[i][1] for i in idx], padding=True,
                                 truncation=True, max_length=self.max_length, return_tensors="pt").to(self.device)
            logits = self.model(**enc).logits.view(-1).float()
            if not torch.isfinite(logits).all():
                logits = self.model.float()(**enc).logits.view(-1).float()
                self.model.half()
            for i, value in zip(idx, logits.tolist()):
                out[i] = value
        return out


def csls(sim: torch.Tensor, k: int, r_cols: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r_rows = sim.topk(min(k, sim.size(1)), dim=1).values.mean(1)
    if r_cols is None:
        r_cols = sim.topk(min(k, sim.size(0)), dim=0).values.mean(0)
    return sim - (r_rows[:, None] + r_cols[None, :]) / 2, r_rows, r_cols


def score_pool(pool: dict, en_text: dict[int, str], vi_text: dict[int, str], encoders: list[BiEncoder],
               reranker: Reranker | None, cfg, out_dir: Path, log) -> dict:
    started = time.time()
    en_ids, vi_ids = pool["en"], pool["vi"]
    en = [en_text[e] for e in en_ids]
    vi = [vi_text[v] for v in vi_ids]
    n, m = len(en), len(vi)
    adjacent = [i for i in range(n - 1) if en_ids[i + 1] == en_ids[i] + 1]
    pair_text = [en[i] + " " + en[i + 1] for i in adjacent]

    per_model = []
    for enc in encoders:
        e, v, p = enc.encode(en), enc.encode(vi), enc.encode(pair_text)
        c11, r_en, r_vi = csls(e @ v.T, cfg.csls_k)
        c21 = csls(p @ v.T, cfg.csls_k, r_vi)[0] if adjacent else None
        per_model.append({"e": e, "r_en": r_en, "r_vi": r_vi, "c11": c11, "c21": c21})
    c11 = torch.stack([x["c11"] for x in per_model]).mean(0)
    cands: dict[tuple, dict] = {}

    def add(kind: str, en_key: tuple, vi_key: tuple, values: list[float]) -> None:
        cands[(kind, en_key, vi_key)] = {"bi_" + enc.name: val for enc, val in zip(encoders, values)}

    c11_cpu = [x["c11"].cpu().tolist() for x in per_model]
    top = c11.topk(min(cfg.k_11, m), dim=1).indices.tolist()
    rev = c11.topk(min(cfg.k_11_rev, n), dim=0).indices.T.tolist()
    pairs11 = {(i, j) for i in range(n) for j in top[i]} | {(i, j) for j in range(m) for i in rev[j]}
    for i, j in sorted(pairs11):
        add("11", (en_ids[i],), (vi_ids[j],), [c[i][j] for c in c11_cpu])
    if adjacent:
        c21 = torch.stack([x["c21"] for x in per_model]).mean(0)
        c21_cpu = [x["c21"].cpu().tolist() for x in per_model]
        top21 = c21.topk(min(cfg.k_21, m), dim=1).indices.tolist()
        for a, i in enumerate(adjacent):
            for j in top21[a]:
                add("21", (en_ids[i], en_ids[i + 1]), (vi_ids[j],), [c[a][j] for c in c21_cpu])

    if m >= 2:
        pool_top = c11.topk(min(cfg.k_12_pool, m), dim=1).indices.tolist()
        combos = list(combinations(range(len(pool_top[0])), 2))
        vp_index: dict[tuple[int, int], int] = {}
        vp_text, J, K, Q = [], [], [], []
        for i in range(n):
            for a, b in combos:
                j, k = pool_top[i][a], pool_top[i][b]
                key = (min(j, k), max(j, k))
                if key not in vp_index:
                    vp_index[key] = len(vp_text)
                    vp_text.append(vi[j] + " " + vi[k])
                J.append(j)
                K.append(k)
                Q.append(vp_index[key])
        device = c11.device
        shape = (n, len(combos))
        J, K, Q = (torch.tensor(t, device=device).view(shape) for t in (J, K, Q))
        rows = torch.arange(n, device=device)[:, None].expand(shape)
        scores12 = []
        for enc, x in zip(encoders, per_model):
            vp = enc.encode(vp_text)
            cos = (vp[Q] * x["e"][rows]).sum(-1)
            scores12.append(cos - (x["r_en"][rows] + (x["r_vi"][J] + x["r_vi"][K]) / 2) / 2)
        s12 = torch.stack(scores12)
        keep = s12.mean(0).topk(min(cfg.k_12, len(combos)), dim=1).indices
        kept = s12.gather(2, keep[None].expand(len(encoders), -1, -1)).permute(1, 2, 0).cpu().tolist()
        keep, J, K = keep.tolist(), J.tolist(), K.tolist()
        for i in range(n):
            for q, values in zip(keep[i], kept[i]):
                j, k = J[i][q], K[i][q]
                add("12", (en_ids[i],), (vi_ids[min(j, k)], vi_ids[max(j, k)]), values)
    bi_time = time.time() - started

    keys = list(cands)
    text_of_en = {e: t for e, t in zip(en_ids, en)}
    text_of_vi = {v: t for v, t in zip(vi_ids, vi)}
    if reranker is not None:
        texts = [(" ".join(text_of_en[e] for e in k[1]), " ".join(text_of_vi[v] for v in k[2])) for k in keys]
        rr = reranker.score(texts)
    else:
        rr = [0.0] * len(keys)

    names = [enc.name for enc in encoders]
    path = out_dir / "candidates.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["kind", "en", "vi"] + [f"bi_{x}" for x in names] + ["bi", "rr"])
        for key, value in zip(keys, rr):
            kind, en_key, vi_key = key
            bis = [cands[key][f"bi_{x}"] for x in names]
            writer.writerow([kind, ";".join(map(str, en_key)), ";".join(map(str, vi_key))]
                            + [f"{b:.5f}" for b in bis] + [f"{sum(bis) / len(bis):.5f}", f"{value:.4f}"])
    counts = {k: sum(key[0] == k for key in keys) for k in ("11", "21", "12")}
    info = {"pool": pool["name"], "en": n, "vi": m, "candidates": counts, "bi_s": round(bi_time, 1),
            "total_s": round(time.time() - started, 1)}
    log(f"[{pool['name']}] {n} EN × {m} VI | candidates {counts} | bi-encoder {bi_time:.0f}s, "
        f"total {info['total_s']:.0f}s ({len(keys) / max(1e-6, info['total_s'] - bi_time):.0f} rerank pairs/s)")
    return info
