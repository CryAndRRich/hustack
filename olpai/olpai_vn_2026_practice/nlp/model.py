from __future__ import annotations

import math

import torch
from torch import nn

from nlp.tokenizer import BOS, EOS, PAD, UNK


def sinusoid(length: int, dim: int) -> torch.Tensor:
    position = torch.arange(length).float().unsqueeze(1)
    freq = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
    table = torch.zeros(length, dim)
    table[:, 0::2] = torch.sin(position * freq)
    table[:, 1::2] = torch.cos(position * freq)
    return table


class Seq2SeqTransformer(nn.Module):
    def __init__(self, src_vocab: int, tgt_vocab: int, d_model: int, heads: int, ffn_dim: int,
                 enc_layers: int, dec_layers: int, dropout: float, max_positions: int):
        super().__init__()
        self.max_positions = max_positions
        self.scale = math.sqrt(d_model)
        self.src_embed = nn.Embedding(src_vocab, d_model, padding_idx=PAD)
        self.tgt_embed = nn.Embedding(tgt_vocab, d_model, padding_idx=PAD)
        for embed in (self.src_embed, self.tgt_embed):
            nn.init.normal_(embed.weight, 0, d_model ** -0.5)
            nn.init.zeros_(embed.weight[PAD])
        self.register_buffer("positions", sinusoid(max_positions + 1, d_model), persistent=False)
        self.drop = nn.Dropout(dropout)
        enc_layer = nn.TransformerEncoderLayer(d_model, heads, ffn_dim, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, enc_layers, norm=nn.LayerNorm(d_model),
                                             enable_nested_tensor=False)
        dec_layer = nn.TransformerDecoderLayer(d_model, heads, ffn_dim, dropout, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, dec_layers, norm=nn.LayerNorm(d_model))
        self.out = nn.Linear(d_model, tgt_vocab, bias=False)
        self.out.weight = self.tgt_embed.weight

    def encode(self, src: torch.Tensor):
        pad = src.eq(PAD)
        x = self.src_embed(src) * self.scale + self.positions[:src.size(1)]
        return self.encoder(self.drop(x), src_key_padding_mask=pad), pad

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, memory_pad: torch.Tensor) -> torch.Tensor:
        length = tgt.size(1)
        causal = torch.triu(torch.ones(length, length, dtype=torch.bool, device=tgt.device), 1)
        x = self.tgt_embed(tgt) * self.scale + self.positions[:length]
        h = self.decoder(self.drop(x), memory, tgt_mask=causal, tgt_key_padding_mask=tgt.eq(PAD),
                         memory_key_padding_mask=memory_pad, tgt_is_causal=True)
        return self.out(h)

    def decode_step(self, token: torch.Tensor, memory: torch.Tensor, memory_pad: torch.Tensor,
                    cache: list[torch.Tensor]) -> tuple[torch.Tensor, list[torch.Tensor]]:
        position = cache[0].size(1) if cache else 0
        x = self.tgt_embed(token[:, None]) * self.scale + self.positions[position]
        new_cache = []
        for i, layer in enumerate(self.decoder.layers):
            h = layer.norm1(x)
            keys = torch.cat([cache[i], h], 1) if cache else h
            new_cache.append(keys)
            x = x + layer.self_attn(h, keys, keys, need_weights=False)[0]
            x = x + layer.multihead_attn(layer.norm2(x), memory, memory, key_padding_mask=memory_pad,
                                         need_weights=False)[0]
            x = x + layer.linear2(layer.activation(layer.linear1(layer.norm3(x))))
        return self.out(self.decoder.norm(x))[:, 0], new_cache

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        memory, pad = self.encode(src)
        return self.decode(tgt_in, memory, pad)


@torch.no_grad()
def check_incremental(model: Seq2SeqTransformer) -> None:
    steps = 6
    training = model.training
    model.eval()
    device = model.out.weight.device
    src = torch.randint(4, model.src_embed.num_embeddings, (3, 9), device=device)
    tgt = torch.randint(4, model.tgt_embed.num_embeddings, (3, steps), device=device)
    tgt[:, 0] = BOS
    memory, pad = model.encode(src)
    full = model.decode(tgt, memory, pad)
    cache = []
    for i in range(steps):
        logits, cache = model.decode_step(tgt[:, i], memory, pad, cache)
        diff = (logits - full[:, i]).abs().max().item()
        if diff > 1e-3 * (1 + full[:, i].abs().max().item()):
            raise RuntimeError(f"decode_step diverges from decode at step {i}: {diff:.2e}")
    model.train(training)


def build_model(cfg, src_vocab: int, tgt_vocab: int) -> Seq2SeqTransformer:
    return Seq2SeqTransformer(src_vocab, tgt_vocab, cfg.d_model, cfg.heads, cfg.ffn_dim, cfg.enc_layers,
                              cfg.dec_layers, cfg.dropout, max_positions=max(cfg.max_tokens * 2, 256))


@torch.inference_mode()
def generate(models, srcs, beam: int, len_penalty: float, max_len_a: float, max_len_b: int, weights=None) -> list:
    if weights is None:
        weights = [1 / len(models)] * len(models)
    if len(weights) != len(models) or abs(sum(weights) - 1) > 1e-6:
        raise ValueError(f"weights must have {len(models)} elements summing to 1; got {weights}")
    device = srcs[0].device
    batch = srcs[0].size(0)
    encoded = [m.encode(s) for m, s in zip(models, srcs)]
    limit = min(m.max_positions for m in models) - 1
    src_len = torch.stack([(~pad).sum(1) for _, pad in encoded]).max(0).values
    max_len = (src_len.float() * max_len_a + max_len_b).long().clamp(1, limit).tolist()
    memories = [(mem.repeat_interleave(beam, 0), pad.repeat_interleave(beam, 0)) for mem, pad in encoded]
    caches = [[] for _ in models]
    tokens = torch.full((batch * beam, 1), BOS, dtype=torch.long, device=device)
    scores = torch.zeros(batch, beam, device=device)
    scores[:, 1:] = float("-inf")
    finished = [[] for _ in range(batch)]
    done = [False] * batch
    neg_inf = float("-inf")

    for step in range(max(max_len)):
        outputs = [m.decode_step(tokens[:, -1], mem, pad, cache)
                   for m, (mem, pad), cache in zip(models, memories, caches)]
        caches = [cache for _, cache in outputs]
        if len(models) == 1:
            logp = outputs[0][0].float().log_softmax(-1)
        else:
            logp = sum(w * logits.float().softmax(-1) for w, (logits, _) in zip(weights, outputs)).log()
        logp[:, [PAD, BOS, UNK]] = neg_inf
        vocab = logp.size(1)
        cand = (scores.view(-1, 1) + logp).view(batch, -1)
        top_scores, top_index = cand.topk(min(2 * beam, cand.size(1)), dim=1)
        top_scores, top_index = top_scores.tolist(), top_index.tolist()
        prefix = None
        rows, next_tokens, next_scores = [], [], []
        for b in range(batch):
            alive = 0
            if not done[b]:
                for score, index in zip(top_scores[b], top_index[b]):
                    if score == neg_inf:
                        break
                    k, token = divmod(index, vocab)
                    length = step + 1
                    if token == EOS or length >= max_len[b]:
                        if prefix is None:
                            prefix = tokens[:, 1:].tolist()
                        seq = prefix[b * beam + k] + ([] if token == EOS else [token])
                        finished[b].append((score / length ** len_penalty, seq))
                        if len(finished[b]) >= beam:
                            break
                    else:
                        rows.append(b * beam + k)
                        next_tokens.append(token)
                        next_scores.append(score)
                        alive += 1
                        if alive == beam:
                            break
                done[b] = len(finished[b]) >= beam or alive == 0
            for _ in range(beam - alive):
                rows.append(b * beam)
                next_tokens.append(PAD)
                next_scores.append(neg_inf)
        if all(done):
            break
        index = torch.tensor(rows, device=device)
        tokens = torch.cat([tokens[index], torch.tensor(next_tokens, device=device).unsqueeze(1)], 1)
        caches = [[layer[index] for layer in cache] for cache in caches]
        scores = torch.tensor(next_scores, device=device).view(batch, beam)

    return [max(f)[1] if f else [] for f in finished]
