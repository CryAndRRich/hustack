import gc
import json
import subprocess
import sys
import threading
from importlib.metadata import PackageNotFoundError, version

import numpy as np
import pandas as pd

from preprocess import (DEADLINE, K_CANDS, LABELS, N_EVID_SUBMIT, N_EVID_VERIFY, NEI, OUT, RERANK_MODEL,
                        VERIFY_MODELS, WORK, clean, covered, find_data_dir, load_claims, load_wiki, log, pip_install,
                        rss_gb, run_workers, title_display, tune_offsets)
from retrieve import HASH, Retriever, retrieval_report

SMOKE_N = {"train": 96, "dev": 48, "test": 32}
SMOKE_PAGES = 100000
RERANK_BUDGET = 75 * 60
EXT_OCNLI = 30000
EXT_FEVER = 20000
SMOKE_EXT = 64
OCNLI_Y = {1: 0, 2: 1, 0: NEI}
FEVER_Y = {"SUPPORTS": 0, "REFUTES": 1, "NOT ENOUGH INFO": NEI}
PREFETCH = {}
PEFT_PREFLIGHT = """
import torch, peft, transformers
from peft import LoraConfig, get_peft_model
class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(4, 4)
        self.score = torch.nn.Linear(4, 3)
    def forward(self, x):
        return self.score(self.q_proj(x))
m = get_peft_model(M(), LoraConfig(r=2, target_modules=["q_proj"], modules_to_save=["score"]))
m(torch.zeros(1, 4)).sum().backward()
print("peft preflight ok: torch", torch.__version__, "transformers", transformers.__version__, "peft", peft.__version__, flush=True)
"""


def ensure_packages():
    from packaging.version import Version
    if Version(version("transformers")) < Version("4.51"):
        pip_install("transformers>=4.51")
    for pkg in ("peft", "opencc"):
        try:
            version(pkg)
        except PackageNotFoundError:
            pip_install(pkg)
    try:
        log(f"uninstalling torchao {version('torchao')}: peft's LoRA dispatcher raises on old torchao")
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=True)
    except PackageNotFoundError:
        pass
    subprocess.run([sys.executable, "-c", PEFT_PREFLIGHT], check=True)


def fever_clean(s):
    for a, b in (("-LRB-", "("), ("-RRB-", ")"), ("-LSB-", "["), ("-RSB-", "]"), ("-COLON-", ":"), ("_", " ")):
        s = s.replace(a, b)
    return s


def prepare_external(path):
    import opencc
    from huggingface_hub import hf_hub_download
    cc = opencc.OpenCC("s2t.json")
    rng = np.random.default_rng(5)
    oc = pd.read_parquet(hf_hub_download("clue/clue", "ocnli/train-00000-of-00001.parquet", repo_type="dataset"))
    oc = oc[oc["label"].isin(list(OCNLI_Y))]
    oc = oc.iloc[rng.permutation(len(oc))[:EXT_OCNLI]]
    rows = [(verifier_text(cc.convert(h), [("", 0, cc.convert(p))]), OCNLI_Y[int(lab)])
            for p, h, lab in zip(oc["sentence1"], oc["sentence2"], oc["label"])]
    n_oc = len(rows)
    with open(hf_hub_download("copenlu/fever_gold_evidence", "train.jsonl", repo_type="dataset"), encoding="utf-8") as f:
        fl = f.readlines()
    for i in rng.permutation(len(fl))[:EXT_FEVER]:
        r = json.loads(fl[i])
        ev = [(fever_clean(e[0]), 0, fever_clean(e[2])) for e in r["evidence"][:N_EVID_VERIFY]]
        rows.append((verifier_text(fever_clean(r["claim"]), ev), FEVER_Y[r["label"]]))
    df = pd.DataFrame(rows, columns=["text", "y"]).sample(frac=1, random_state=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    PREFETCH["external"] = f"{n_oc} OCNLI (s2t) + {len(df) - n_oc} FEVER rows, labels {np.bincount(df['y'], minlength=3).tolist()}"


def prefetch():
    try:
        from huggingface_hub import snapshot_download
        for m in (RERANK_MODEL, VERIFY_MODELS[0]):
            snapshot_download(m, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken"])
        prepare_external(WORK / "external.parquet")
    except Exception as e:
        PREFETCH["error"] = repr(e)


def smoke_subset(claims, wiki):
    sub = pd.concat([claims[claims["split"] == s].sample(n, random_state=0) for s, n in SMOKE_N.items()])
    titles, texts, lines = wiki
    t2i = {t: i for i, t in enumerate(titles)}
    keep = sorted(set(range(SMOKE_PAGES)) | {t2i[t] for gs in sub["groups"] for g in gs for t, _ in g})
    return sub, ([titles[i] for i in keep], [texts[i] for i in keep], [lines[i] for i in keep])


def retrieve_stage(claims, wiki, work):
    work.mkdir(parents=True, exist_ok=True)
    retr = Retriever(*wiki)
    log("retriever built")
    cands, pools = retr.run(claims)
    retrieval_report(claims, cands, pools, retr.titles)
    gold = retr.gold_texts(claims[claims["split"] != "test"])
    retr.WT = retr.base2idx = None
    del pools
    gc.collect()
    c = claims.reset_index()[["qid", "id", "split", "claim", "y"]]
    c.to_parquet(work / "claims.parquet")
    cands.to_parquet(work / "cands.parquet")
    gold.to_parquet(work / "gold.parquet")
    log(f"retrieval written: {len(cands)} candidate rows, {len(gold)} gold sentences, peak RSS {rss_gb():.1f} GB")
    return retr


def verifier_text(claim, ev):
    return f"主張：{claim[:160]}\n" + "".join(
        f"證據{j}{'（' + title_display(t) + '）' if t else ''}：{x[:160]}\n" for j, (t, _, x) in enumerate(ev, 1))


def build_verifier_inputs(claims, top, gold_text, rng):
    tr, inf = [], []
    for qid, row in claims.iterrows():
        ev = list(top.get(qid, []))[:N_EVID_VERIFY]
        if row["split"] == "train":
            if row["y"] != NEI and not covered({(t, s) for t, s, _ in ev}, row["groups"]):
                g = min(row["groups"], key=len)[:N_EVID_VERIFY]
                need = [(t, s, gold_text.get((t, s), "")) for t, s in g if (t, s) not in {(a, b) for a, b, _ in ev}]
                gset = set(g)
                while len(ev) + len(need) > N_EVID_VERIFY:
                    drop = max(i for i, e in enumerate(ev) if (e[0], e[1]) not in gset)
                    ev.pop(drop)
                for e in need:
                    ev.insert(int(rng.integers(0, len(ev) + 1)), e)
            tr.append((verifier_text(row["claim"], ev), int(row["y"])))
        else:
            inf.append((qid, verifier_text(row["claim"], ev)))
    return pd.DataFrame(tr, columns=["text", "y"]), pd.DataFrame(inf, columns=["qid", "text"])


def merge_scores(n, work, pattern):
    score = np.full(n, np.nan, np.float32)
    for f in work.glob(pattern):
        z = np.load(f)
        score[z["idx"]] = z["score"]
    return score


def load_probs(work, pattern):
    out = {}
    for f in work.glob(pattern):
        z = np.load(f)
        out.update(zip(z["qid"].tolist(), z["probs"]))
    return out


def dev_hit(cands, dev_sr, k):
    d = cands[cands["qid"].isin(dev_sr.index)]
    keys = {q: set(list(zip(g["title"], g["sid"]))[:k]) for q, g in d.groupby("qid", sort=False)}
    return np.mean([covered(keys.get(q, set()), gs) for q, gs in dev_sr["groups"].items()])


def ev_features(qids, top, top_score):
    rows = []
    for q in qids:
        s = top_score.get(q, np.zeros(0, np.float32))
        s = np.concatenate([s, np.full(4 - len(s), (s[-1] if len(s) else 0.0) - 5.0)])
        rows.append([*s, s[2] - s[3], s[0] - s[2], len({t for t, *_ in top.get(q, [])[:N_EVID_SUBMIT]})])
    return np.array(rows, dtype=np.float64)


def fit_predict_q(X, y, Xp):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if len(y) == 0 or y.min() == y.max():
        return np.full(len(Xp), y.mean() if len(y) else 0.5)
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X, y).predict_proba(Xp)[:, 1]


def evidence_q(Xd, ok, sr, Xt):
    from sklearn.model_selection import KFold
    q_dev = np.zeros(len(Xd))
    for tr_i, va_i in KFold(5, shuffle=True, random_state=0).split(Xd):
        f = tr_i[sr[tr_i]]
        q_dev[va_i] = fit_predict_q(Xd[f], ok[f], Xd[va_i])
    return np.clip(q_dev, 0.01, 0.99), np.clip(fit_predict_q(Xd[sr], ok[sr], Xt), 0.01, 0.99)


def evidence_add(q):
    lq = np.log(q)
    return np.stack([lq, lq, np.zeros_like(lq)], 1)


def test_diagnostics(claims, top, probs):
    from sklearn.preprocessing import normalize
    split = claims["split"].to_numpy()
    Q = normalize(HASH.transform([clean(c) for c in claims["claim"]])).tocsr()
    T = Q[split == "train"].T.tocsc()
    gold_pages = {t for gs in claims.loc[split == "train", "groups"] for g in gs for t, _ in g}
    new = claims["id"].to_numpy() >= 32000
    for name, msk in (("dev", split == "dev"), ("test original", (split == "test") & ~new),
                      ("test new", (split == "test") & new)):
        idx = np.where(msk)[0]
        if not len(idx):
            continue
        sim = np.concatenate([(Q[idx[a:a + 1000]] @ T).max(axis=1).toarray().ravel() for a in range(0, len(idx), 1000)])
        qids = claims.index[idx]
        page = np.mean([bool(top.get(q)) and top[q][0][0] in gold_pages for q in qids])
        conf = np.mean([probs[q].max() for q in qids]) if probs else float("nan")
        log(f"diagnostics {name}: n={len(idx)} nearest-train-claim cosine mean {sim.mean():.3f} share>0.8 "
            f"{np.mean(sim > 0.8):.3f}  top-1 page is a train gold page {page:.3f}  verifier max-prob {conf:.3f}")


def gpu_stages(claims, retr, work, data_dir, ngpu, smoke):
    PREFETCH["thread"].join()
    if "error" in PREFETCH:
        raise RuntimeError("download failed (internet on?): " + PREFETCH["error"])
    log(f"external data: {PREFETCH['external']}")
    for pat in ("rr_rank*.npz", "hop_rank*.npz", "vp_rank*.npz", "vp1_rank*.npz"):
        for f in work.glob(pat):
            f.unlink()
    split = claims["split"]
    run_workers("rerank_worker.py", ["--work", work, "--model", RERANK_MODEL, "--train_budget",
                                     60 if smoke else RERANK_BUDGET, "--deadline", DEADLINE - 6 * 3600,
                                     "--max_steps", 8 if smoke else 0], ngpu)
    cands = pd.read_parquet(work / "cands.parquet")
    score = merge_scores(len(cands), work, "rr_rank*.npz")
    is_dt = cands["qid"].map(split).isin(["dev", "test"]).to_numpy()
    assert not np.isnan(score[is_dt]).any(), "dev/test candidates left unscored"
    log(f"reranker: {np.isnan(score).sum()} train rows unscored (prefilter rank >= 24 or deadline; ranked below scored rows)")
    cands["score"] = np.where(np.isnan(score), cands["pre"] - 1e3, score)
    cands = cands.sort_values(["qid", "score"], ascending=[True, False])

    hop = retr.hop_candidates(cands[cands["qid"].map(split).isin(["dev", "test"])])
    hop["claim"] = hop["qid"].map(claims["claim"])
    hop.to_parquet(work / "hop.parquet")
    run_workers("rerank_worker.py", ["--work", work, "--model", work / "reranker", "--score_file", "hop.parquet"], ngpu)
    hs = merge_scores(len(hop), work, "hop_rank*.npz")
    assert not np.isnan(hs).any(), "hop candidates left unscored"
    hop = hop.drop(columns="claim").assign(pre=np.nan, rank=-1, score=hs)
    with_hop = pd.concat([cands, hop], ignore_index=True).sort_values(["qid", "score"], ascending=[True, False])
    lab = claims[split != "test"]
    dev_sr = lab[(lab["split"] == "dev") & (lab["y"] != NEI)]
    h = {(k, w): dev_hit(c, dev_sr, k) for k in (N_EVID_SUBMIT, N_EVID_VERIFY) for w, c in ((0, cands), (1, with_hop))}
    use_hop = h[(N_EVID_SUBMIT, 1)] > h[(N_EVID_SUBMIT, 0)]
    log(f"hop: {len(hop)} linked-page rows for dev/test; dev gold group within top-{N_EVID_SUBMIT} "
        f"{h[(N_EVID_SUBMIT, 0)]:.4f} -> {h[(N_EVID_SUBMIT, 1)]:.4f}, top-{N_EVID_VERIFY} {h[(N_EVID_VERIFY, 0)]:.4f} -> "
        f"{h[(N_EVID_VERIFY, 1)]:.4f}; using hop rows: {use_hop}")
    if use_hop:
        cands = with_hop
    del with_hop, hop

    top = {q: list(zip(d["title"], d["sid"], d["text"])) for q, d in cands.groupby("qid", sort=False)}
    gold = pd.read_parquet(work / "gold.parquet")
    gold_text = dict(zip(zip(gold["title"], gold["sid"]), gold["text"]))
    cands = cands[cands["qid"].map(split).isin(["dev", "test"])][["qid", "title", "sid", "pre", "score"]]
    top_score = {q: d["score"].to_numpy()[:4] for q, d in cands.groupby("qid", sort=False)}
    del gold
    gc.collect()

    ev_ok = {q: covered({(t, s) for t, s, _ in top.get(q, [])[:N_EVID_SUBMIT]}, gs) for q, gs in lab["groups"].items()}
    for sname, d in lab[lab["y"] != NEI].groupby("split"):
        log(f"reranker {sname}: gold group within top-{N_EVID_SUBMIT} {np.mean([ev_ok[q] for q in d.index]):.4f}  "
            f"top-{N_EVID_VERIFY} {np.mean([covered({(t, s) for t, s, _ in top.get(q, [])[:N_EVID_VERIFY]}, d.at[q, 'groups']) for q in d.index]):.4f}")

    vtr, vinf = build_verifier_inputs(claims, top, gold_text, np.random.default_rng(4))
    vtr.to_parquet(work / "vtrain.parquet")
    vinf.to_parquet(work / "vinfer.parquet")
    log(f"verifier inputs: train {len(vtr)} (labels {np.bincount(vtr['y'], minlength=3).tolist()}), infer {len(vinf)}, peak RSS {rss_gb():.1f} GB")
    try:
        run_workers("verify_worker.py", ["--work", work, "--models", ",".join(VERIFY_MODELS),
                                         "--ext", WORK / "external.parquet", "--ext_rows", SMOKE_EXT if smoke else 0,
                                         "--deadline", DEADLINE - 25 * 60, "--max_steps", 8 if smoke else 0], ngpu)
    except Exception as e:
        if smoke:
            raise
        log("VERIFIER FAILED, using whatever probabilities were saved:", repr(e))
    sets = {}
    for name, pat in (("final", "vp_rank*.npz"), ("epoch1", "vp1_rank*.npz")):
        pr = load_probs(work, pat)
        if pr and set(pr) == set(vinf["qid"]):
            sets[name] = pr
        elif pr:
            log(f"verifier {name} probabilities incomplete, ignored")
    assert not smoke or set(sets) == {"final", "epoch1"}, "smoke must produce final and epoch-1 probabilities"
    if not sets:
        log("no verifier probabilities, falling back to all-NEI labels")

    dev = claims[split == "dev"]
    test = claims[split == "test"]
    pred_test = np.full(len(test), NEI)
    choice = {}
    if sets:
        y = dev["y"].to_numpy()
        ok = np.array([ev_ok[q] for q in dev.index])
        sr = y != NEI
        q_dev, q_test = evidence_q(ev_features(dev.index, top, top_score), ok, sr,
                                   ev_features(test.index, top, top_score))
        log(f"evidence model (5-fold on dev): mean q on S/R hits {q_dev[sr & ok].mean():.3f}, "
            f"on misses {q_dev[sr & ~ok].mean() if (sr & ~ok).any() else float('nan'):.3f}")
        best = None
        for name, pr in sets.items():
            logp = np.log(np.stack([pr[q] for q in dev.index]) + 1e-9)
            raw = np.mean((logp.argmax(1) == y) & (~sr | ok))
            for rule, add in (("offsets", 0.0), ("offsets+evidence", evidence_add(q_dev))):
                f, bs, br = tune_offsets(logp + add, y, ok)
                log(f"dev {name} {rule}: label acc {np.mean(logp.argmax(1) == y):.4f}  FEVER raw {raw:.4f}  "
                    f"tuned {f:.4f} (offsets S {bs:+.1f} R {br:+.1f})  n={len(dev)}")
                if best is None or f > best[0] + 1e-9:
                    best = (f, name, rule, bs, br)
        f, name, rule, bs, br = best
        choice = {"probs": name, "rule": rule, "S": bs, "R": br, "dev_fever": f}
        logpt = np.log(np.stack([sets[name][q] for q in test.index]) + 1e-9)
        add_t = evidence_add(q_test) if rule == "offsets+evidence" else 0.0
        pred_test = np.argmax(logpt + add_t + np.array([bs, br, 0.0]), 1)
        log(f"chosen: {name} probabilities, rule {rule}, dev FEVER {f:.4f}")
    new_block = test["id"].to_numpy() >= 32000
    for nm, msk in (("original-test ids", ~new_block), ("new ids >= 32000", new_block)):
        log(f"test {nm}: n={msk.sum()} label shares {np.round(np.bincount(pred_test[msk], minlength=3) / max(msk.sum(), 1), 3).tolist()}")
    test_diagnostics(claims, top, sets.get("final"))

    sample = pd.read_csv(data_dir / "sample_submission.csv")
    lab_of = dict(zip(test["id"], pred_test))
    q_of = dict(zip(test["id"], test.index))
    sub = pd.DataFrame({"id": sample["id"]})
    sub["predicted_label"] = [LABELS[lab_of.get(i, NEI)] for i in sub["id"]]
    sub["predicted_evidence"] = [json.dumps([[t, int(s)] for t, s, _ in top.get(q_of.get(i), [])[:N_EVID_SUBMIT]],
                                            ensure_ascii=False) for i in sub["id"]]
    assert len(sub) == 6000 and sub["id"].is_unique and set(test["id"]) <= set(sub["id"])
    assert smoke or len(test) == len(sub)
    assert sub["predicted_label"].isin(LABELS).all()
    assert all(len(json.loads(e)) <= N_EVID_SUBMIT for e in sub["predicted_evidence"])
    if smoke:
        sub.to_csv(work / "submission_smoke.csv", index=False)
        log("smoke ok")
        return
    sub.to_csv(OUT / "submission.csv", index=False)
    extra = OUT / "extra"
    extra.mkdir(exist_ok=True)
    cands.to_parquet(extra / "rerank_dev_test.parquet")
    if sets:
        qs = list(dev.index) + list(test.index)
        pr = pd.DataFrame({"qid": qs, "q_evidence": np.concatenate([q_dev, q_test])})
        for name, d in sets.items():
            arr = np.stack([d[q] for q in qs])
            for j, col in enumerate(("sup", "ref", "nei")):
                pr[f"{name}_{col}"] = arr[:, j]
        pr.merge(claims.reset_index()[["qid", "id", "split", "y"]], on="qid").to_parquet(extra / "verifier_probs.parquet")
    (extra / "choice.json").write_text(json.dumps(choice, default=float))
    log(f"evidence count per test claim: {np.bincount([len(json.loads(e)) for e in sub['predicted_evidence']]).tolist()}")
    log(f"SUBMIT: {OUT / 'submission.csv'}")


if __name__ == "__main__":
    import torch
    ngpu = torch.cuda.device_count()
    assert ngpu >= 1, "GPU accelerator required (T4 x2)"
    log(f"{ngpu} GPU(s), K_CANDS {K_CANDS}")
    ensure_packages()
    PREFETCH["thread"] = threading.Thread(target=prefetch, daemon=True)
    PREFETCH["thread"].start()
    data_dir = find_data_dir()
    claims = load_claims(data_dir)
    claims.index.name = "qid"
    wiki = load_wiki(data_dir / "wiki_pages.jsonl")
    log(f"data: {len(claims)} claims, {len(wiki[0])} pages, data dir {data_dir}")

    sm_claims, sm_wiki = smoke_subset(claims, wiki)
    sm_retr = retrieve_stage(sm_claims, sm_wiki, WORK / "smoke")
    del sm_wiki
    gpu_stages(sm_claims, sm_retr, WORK / "smoke", data_dir, ngpu, smoke=True)
    del sm_retr

    retr = retrieve_stage(claims, wiki, WORK / "full")
    del wiki
    gc.collect()
    gpu_stages(claims, retr, WORK / "full", data_dir, ngpu, smoke=False)
    log("done")
