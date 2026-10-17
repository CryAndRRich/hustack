import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from preprocess import EDU_COLS, EDU_DESC_COLS, NO_DESC, NO_EDU, get_out_dir

EMB_MODEL = "cointegrated/rubert-tiny2"
EMB_MAX_LEN = 512
EMB_BATCH = 32
EMB_FIELDS = ["description", "experience_desc", "education"]
EMB_DECIMALS = 4
TFIDF_CHAR = dict(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=60000, sublinear_tf=True)
TFIDF_WORD = dict(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=40000, sublinear_tf=True,
                  token_pattern=r"(?u)\b\w[\w+#]*")


def emb_texts(df):
    desc = df["description"].fillna("").astype(str).str.strip()
    desc = desc.where(~desc.str.contains(NO_DESC, case=False, regex=False), "")
    exp = df["experience_desc"].fillna("").astype(str).str.replace("_x000D_", " ", regex=False).str.strip()
    parts = []
    for e, d in zip(EDU_COLS, EDU_DESC_COLS):
        name = df[e].fillna("").astype(str).str.strip()
        name = name.where(~name.str.contains(NO_EDU, case=False, regex=False), "")
        parts.append((name + ". " + df[d].fillna("").astype(str).str.strip()).str.strip(" ."))
    edu = pd.concat(parts, axis=1).agg(lambda r: " | ".join(x for x in r if x), axis=1)
    return {"description": desc, "experience_desc": exp, "education": edu}


def find_model():
    if Path("/kaggle/input").exists():
        for cfg in sorted(Path("/kaggle/input").rglob("config.json")):
            if "rubert-tiny2" in str(cfg.parent).lower():
                return str(cfg.parent)
    return EMB_MODEL


def embed_unique(texts, tok, model):
    import torch
    uniq = sorted({t for t in texts if t}, key=len)
    vecs = {}
    with torch.no_grad():
        for i in range(0, len(uniq), EMB_BATCH):
            batch = uniq[i:i + EMB_BATCH]
            enc = tok(batch, padding=True, truncation=True, max_length=EMB_MAX_LEN, return_tensors="pt")
            cls = model(**enc).last_hidden_state[:, 0]
            vecs.update(zip(batch, torch.nn.functional.normalize(cls).numpy()))
    dim = model.config.hidden_size
    return np.stack([vecs.get(t, np.zeros(dim, dtype=np.float32)) for t in texts]), len(uniq)


def embed_frames(train, test):
    cache = get_out_dir() / "emb_cache.npz"
    if cache.exists():
        z = np.load(cache)
        if z["E_tr"].shape[0] == len(train) and z["E_te"].shape[0] == len(test):
            print(f"embeddings loaded from {cache}")
            return z["E_tr"], z["E_te"]
    try:
        from transformers import AutoModel, AutoTokenizer
        src = find_model()
        tok, model = AutoTokenizer.from_pretrained(src), AutoModel.from_pretrained(src).eval()
    except Exception as e:
        raise RuntimeError(f"cannot load {EMB_MODEL}: turn Internet ON or attach the model as a dataset ({e})")
    print(f"embedding model: {src} (dim {model.config.hidden_size})")
    t_tr, t_te = emb_texts(train), emb_texts(test)
    blocks_tr, blocks_te = [], []
    for f in EMB_FIELDS:
        both = list(t_tr[f]) + list(t_te[f])
        E, n_uniq = embed_unique(both, tok, model)
        print(f"  {f}: {n_uniq} unique non-empty texts, empty share {np.mean([not t for t in both]):.3f}")
        blocks_tr.append(E[:len(train)])
        blocks_te.append(E[len(train):])
    E_tr = np.round(np.hstack(blocks_tr), EMB_DECIMALS).astype(np.float32)
    E_te = np.round(np.hstack(blocks_te), EMB_DECIMALS).astype(np.float32)
    np.savez_compressed(cache, E_tr=E_tr, E_te=E_te)
    print(f"embedding checksum (rounded to {EMB_DECIMALS} dp): {emb_checksum(E_tr, E_te)}")
    return E_tr, E_te


def emb_checksum(E_tr, E_te):
    return hashlib.md5(np.ascontiguousarray(np.vstack([E_tr, E_te])).tobytes()).hexdigest()[:12]


def tfidf_frames(train, test):
    t_tr, t_te = emb_texts(train), emb_texts(test)
    doc_tr = (t_tr["description"] + " \n " + t_tr["experience_desc"] + " \n " + t_tr["education"]).str.lower()
    doc_te = (t_te["description"] + " \n " + t_te["experience_desc"] + " \n " + t_te["education"]).str.lower()
    both = pd.concat([doc_tr, doc_te]).str.replace("ё", "е", regex=False)
    mats = [TfidfVectorizer(**cfg).fit_transform(both) for cfg in (TFIDF_CHAR, TFIDF_WORD)]
    M = sparse.hstack(mats).tocsr()
    print(f"tf-idf: {M.shape[1]} columns (char {mats[0].shape[1]}, word {mats[1].shape[1]})")
    return M[:len(train)], M[len(train):]


ST_MODEL = "deepvk/USER-bge-m3"
ST_MAX_LEN = 512
ST_BATCH = 32


def embed_st(train, test, model_name=ST_MODEL):
    cache = get_out_dir() / "emb_st_cache.npz"
    if cache.exists():
        z = np.load(cache)
        if z["E_tr"].shape[0] == len(train) and z["E_te"].shape[0] == len(test):
            print(f"{model_name} embeddings loaded from {cache}")
            return z["E_tr"], z["E_te"], z["D_tr"], z["D_te"]
    import torch
    from sentence_transformers import SentenceTransformer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    model.max_seq_length = ST_MAX_LEN
    t_tr, t_te = emb_texts(train), emb_texts(test)
    blocks = []
    for f in EMB_FIELDS:
        both = list(t_tr[f]) + list(t_te[f])
        uniq = sorted({t for t in both if t}, key=len)
        vecs = model.encode(uniq, batch_size=ST_BATCH, normalize_embeddings=True, show_progress_bar=False)
        lut = dict(zip(uniq, vecs))
        dim = vecs.shape[1]
        blocks.append(np.stack([lut.get(t, np.zeros(dim, dtype=np.float32)) for t in both]))
        print(f"  {model_name} {f}: {len(uniq)} unique texts on {device}")
    doc = blocks[0] + blocks[1]
    norm = np.linalg.norm(doc, axis=1, keepdims=True)
    doc = np.where(norm > 0, doc / np.maximum(norm, 1e-12), 0.0)
    E = np.round(np.hstack(blocks), EMB_DECIMALS).astype(np.float32)
    D = np.round(doc, EMB_DECIMALS).astype(np.float32)
    n = len(train)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    np.savez_compressed(cache, E_tr=E[:n], E_te=E[n:], D_tr=D[:n], D_te=D[n:])
    return E[:n], E[n:], D[:n], D[n:]
