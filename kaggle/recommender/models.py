import copy

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from numpy.lib.stride_tricks import sliding_window_view
from torch import nn

from content import YEAR, era_window, gram, transitions
from preprocess import N_ITEMS, matrix

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEQ_LEN = 200
FUTURE = 20
BATCH = 64
PRE_BATCH = 128
ERA_YEARS = 2


def active(X):
    return np.asarray(X.sum(0)).ravel() > 0


def top_k_per_row(W, k):
    if k < W.shape[1]:
        kth = np.partition(W, -k, axis=1)[:, -k][:, None]
        W = np.where(W >= kth, W, 0)
    return W.astype(np.float32)


def cosine(X, shrink=0):
    G = (X.T @ X).toarray().astype(np.float32)
    norm = np.sqrt(np.diag(G))
    S = G / (np.outer(norm, norm) + shrink + 1e-9)
    np.fill_diagonal(S, 0)
    return S


def _embed(B, act):
    W = np.zeros((N_ITEMS, N_ITEMS), np.float32)
    W[np.ix_(act, act)] = B
    return W


def _ease_compact(G, act, lam, GY=None, drop=0.0):
    Ga = G[np.ix_(act, act)]
    P = np.linalg.inv(Ga + np.diag(lam + drop / (1 - drop) * np.diag(Ga)))
    if GY is None:
        B = P / -np.diag(P)
        np.fill_diagonal(B, 0)
    else:
        B = P @ GY[np.ix_(act, act)]
        B -= P * (np.diag(B) / np.diag(P))[None, :]
    return B.astype(np.float32)


def _trans_compact(T, act, cnt, k, beta):
    idx = np.flatnonzero(act)
    T = T[idx][:, idx].toarray()
    T /= np.maximum(T.sum(1, keepdims=True), 1e-9)
    T /= np.maximum(cnt[act], 1)[None, :] ** beta
    np.fill_diagonal(T, 0)
    return top_k_per_row(T, k)


def ease(ctx, lam):
    X = ctx["X"]
    act = active(X)
    return _embed(_ease_compact((X.T @ X).toarray().astype(np.float64), act, lam), act)


def rp3(ctx, alpha, beta, k):
    X = ctx["X"]
    deg_u, deg_i = np.asarray(X.sum(1)).ravel(), np.asarray(X.sum(0)).ravel()
    Pui = X.multiply(1 / np.maximum(deg_u, 1)[:, None]).power(alpha).tocsr()
    Piu = X.T.multiply(1 / np.maximum(deg_i, 1)[:, None]).power(alpha).tocsr()
    W = (Piu @ Pui).toarray() / np.maximum(deg_i, 1)[None, :] ** beta
    np.fill_diagonal(W, 0)
    return top_k_per_row(W, k)


def content(ctx, k):
    S = ctx["E"] @ ctx["E"].T
    np.fill_diagonal(S, 0)
    return top_k_per_row(S, k)


def ease_ext(ctx, lam):
    e = ctx["ext"]
    act = e["cnt"] > 0
    return _embed(_ease_compact(e["G"], act, lam), act)


def ease_like_ext(ctx, lam):
    e = ctx["ext"]
    act = e["cnt"] > 0
    return _embed(_ease_compact(e["G"], act, lam, e["GY"]), act)


def edlae_ext(ctx, lam, drop):
    e = ctx["ext"]
    act = e["cnt"] > 0
    return _embed(_ease_compact(e["G"], act, lam, drop=drop), act)


def trans(ctx, k, beta, steps, session):
    e = ctx["ext"]
    act = e["cnt"] > 0
    return _embed(_trans_compact(transitions(e["seq"], steps, session), act, e["cnt"], k, beta), act)


GENERATORS = {"ease": ease, "rp3": rp3, "content": content, "ease_ext": ease_ext, "ease_like_ext": ease_like_ext,
              "edlae_ext": edlae_ext, "trans": trans}
STATIC = {"ease_ext", "ease_like_ext", "edlae_ext", "trans"}


def timepop(ctx, weeks):
    C, w = ctx["ext"]["cum_like"], ctx["week"]
    hi = np.clip(w + weeks, 0, C.shape[1] - 1)
    return (C[:, hi] - C[:, np.clip(w, 0, C.shape[1] - 1)]).T


def eras(ctx):
    year = (1970 + (ctx["ext"]["t0"] + ctx["week"] * 7 * 86400) / YEAR).astype(int)
    b = (year - 1995) // ERA_YEARS
    return {int(k): np.flatnonzero(b == k) for k in np.unique(b)}


def _era_model(ctx, name, params, b, fit):
    key = (name, tuple(sorted(params.items())), b)
    cache = ctx["ext"]["cache"]
    if key not in cache:
        cache[key] = fit(era_window(ctx["ext"], 1995 + ERA_YEARS * b - 1, 1995 + ERA_YEARS * (b + 1) + 1))
    return cache[key]


def _era_scores(ctx, name, params, tau, fit):
    Xt = matrix(ctx["df"], ctx["n_users"], tau)
    out = np.zeros((ctx["n_users"], N_ITEMS), np.float32)
    for b, users in eras(ctx).items():
        act, B = _era_model(ctx, name, params, b, fit)
        out[np.ix_(users, act)] = Xt[users][:, act] @ B
    return out


def ease_like_era(ctx, lam, tau):
    e = ctx["ext"]

    def fit(m):
        s = e["seq"]
        users = np.unique(s["user"][m], return_inverse=True)[1]
        X = sp.csr_matrix((np.ones(m.sum(), np.float32), (users, s["item"][m])), shape=(users.max() + 1, N_ITEMS))
        Y = sp.csr_matrix((s["like"][m].astype(np.float32), (users, s["item"][m])), shape=X.shape)
        Y.eliminate_zeros()
        G, GY = gram(X, Y)
        act = np.diag(G) > 0
        return np.flatnonzero(act), _ease_compact(G, act, lam * m.sum() / len(m), GY)

    return _era_scores(ctx, "ease_like_era", {"lam": lam}, tau, fit)


def trans_era(ctx, k, steps, session, tau):
    e = ctx["ext"]

    def fit(m):
        T = transitions(e["seq"], steps, session, m)
        cnt = np.bincount(e["seq"]["item"][m], minlength=N_ITEMS).astype(np.float32)
        act = cnt > 0
        return np.flatnonzero(act), _trans_compact(T, act, cnt, k, 0.0)

    return _era_scores(ctx, "trans_era", {"k": k, "steps": steps, "session": session}, tau, fit)


USER_GENERATORS = {"timepop": timepop, "ease_like_era": ease_like_era, "trans_era": trans_era}
ERA_GENERATORS = {"ease_like_era", "trans_era"}


class SASRec(nn.Module):
    def __init__(self, E, d=128, layers=2, heads=2, drop=0.2):
        super().__init__()
        self.register_buffer("content", torch.cat([torch.zeros(1, E.shape[1]), torch.from_numpy(E)]))
        self.item = nn.Embedding(N_ITEMS + 1, d, padding_idx=0)
        self.proj = nn.Linear(E.shape[1], d, bias=False)
        self.pos = nn.Embedding(SEQ_LEN, d)
        self.drop = nn.Dropout(drop)
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, drop, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)

    def forward(self, seq):
        W = self.item.weight + self.proj(self.content)
        L = seq.shape[1]
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=seq.device), 1)
        h = self.norm(self.enc(self.drop(F.embedding(seq, W) + self.pos.weight[:L]), mask=causal))
        return h, W[1:]


def windows(seqs, rng):
    inp = np.zeros((len(seqs), SEQ_LEN), np.int64)
    tgt = np.full((len(seqs), SEQ_LEN, FUTURE), -1, np.int64)
    for r, s in enumerate(seqs):
        b = len(s) if rng.random() < 0.5 else rng.integers(1, len(s) + 1)
        a = max(0, b - SEQ_LEN)
        inp[r, :b - a] = s[a:b]
        tgt[r, :b - a] = sliding_window_view(np.r_[s, np.zeros(FUTURE, s.dtype)][a + 1:b + FUTURE], FUTURE) - 1
    return inp, tgt


def train_epoch(model, opt, scaler, seqs, rng, batch):
    model.train()
    for b in np.array_split(rng.permutation(len(seqs)), max(1, len(seqs) // batch)):
        inp, tgt = (torch.from_numpy(a).to(DEVICE) for a in windows([seqs[j] for j in b], rng))
        with torch.autocast(DEVICE, dtype=torch.float16, enabled=DEVICE == "cuda"):
            h, W = model(inp)
            keep = (tgt >= 0).any(-1)
            logits = h[keep] @ W.T
        t = tgt[keep]
        lsm = F.log_softmax(logits.float(), -1)
        loss = -(lsm.gather(1, t.clamp(min=0)) * (t >= 0)).sum() / (t >= 0).sum()
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
    return loss.item()


@torch.no_grad()
def predict(model, hist, n_users):
    model.eval()
    out = np.zeros((n_users, N_ITEMS), np.float32)
    users = np.array(list(hist))
    for b in np.array_split(users, max(1, len(users) // 128)):
        inp = np.zeros((len(b), SEQ_LEN), np.int64)
        last = [min(len(hist[u]), SEQ_LEN) - 1 for u in b]
        for r, u in enumerate(b):
            inp[r, :last[r] + 1] = hist[u][-SEQ_LEN:]
        h, W = model(torch.from_numpy(inp).to(DEVICE))
        out[b] = (h[torch.arange(len(b)), torch.tensor(last)].float() @ W.float().T).cpu().numpy()
    return out


def pretrain(ext, E, epochs, max_users=None, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    s = ext["seq"]
    cuts = np.flatnonzero(np.diff(s["user"])) + 1
    seqs = [q + 1 for q in np.split(s["item"], cuts) if len(q) >= 2][:max_users]
    model = SASRec(E).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scaler = torch.amp.GradScaler(enabled=DEVICE == "cuda")
    for ep in range(1, epochs + 1):
        loss = train_epoch(model, opt, scaler, seqs, rng, PRE_BATCH)
        print(f"    sasrec pretrain epoch {ep}/{epochs} on {len(seqs)} sequences: loss {loss:.3f}", flush=True)
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def finetune(state, df, n_users, E, checkpoints, seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    hist = {u: g.movieId.to_numpy() + 1 for u, g in df.groupby("userId")}
    seqs = list(hist.values())
    model = SASRec(E).to(DEVICE)
    model.load_state_dict(copy.deepcopy(state))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scaler = torch.amp.GradScaler(enabled=DEVICE == "cuda")
    out, ep = {}, 0
    for c in sorted(checkpoints):
        while ep < c:
            train_epoch(model, opt, scaler, seqs, rng, BATCH)
            ep += 1
        out[c] = predict(model, hist, n_users)
    return out
