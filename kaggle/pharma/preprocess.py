import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

T0 = time.time()
FIELDS = ["Mechanism", "Pharmacodynamics", "Metabolism", "Half_Life", "Absorption", "Toxicity",
          "Protein_Binding", "Elimination_Route", "Indication", "Warning", "CYP450_Enzymes"]
TEXT_FIELDS = ["Mechanism", "Pharmacodynamics", "Metabolism", "Absorption", "Toxicity",
               "Elimination_Route", "Indication", "Warning"]
KW = ["boxed", "torsade", "QT", "CYP3A4 inhibitor", "strong inhibitor", "inducer", "anticoagul",
      "bleeding", "warfarin", "hepatotox", "nephrotox", "serotonin", "MAO", "potassium", "hyperkal",
      "hypotension", "CNS depress", "respiratory depression", "myopathy", "rhabdomyolysis", "seizure",
      "contraindicat", "opioid", "benzodiazepine", "P-glycoprotein", "narrow therapeutic",
      "P-gp", "renal impairment", "hepatic impairment", "additive", "synerg"]
CYP_RE = re.compile(r"CYP\s?(\d[A-Z]\d{1,2})", re.I)
SEV_CLASSES = ["Minor", "Moderate", "Major"]
TRAIN_BP = (4086, 14654)
TEST_BP = (1090, 3846)


def set_breakpoints(train_bp, test_bp):
    global TRAIN_BP, TEST_BP
    TRAIN_BP, TEST_BP = tuple(train_bp), tuple(test_bp)
    log(f"breakpoints set to train {TRAIN_BP} test {TEST_BP}")


def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)


def find_data():
    roots = [Path("/kaggle/input")]
    try:
        roots.append(Path(__file__).resolve().parent.parent / "data")
    except NameError:
        pass
    roots.append(Path.cwd() / "data")
    for r in roots:
        if not r.exists():
            continue
        for dp, _, fn in os.walk(r):
            if "train.csv" in fn and "test.csv" in fn:
                if "Target_PRR_Nausea" in pd.read_csv(Path(dp) / "train.csv", nrows=1).columns:
                    return Path(dp)
    raise FileNotFoundError("competition data (train.csv with Target_PRR_Nausea) not found")


def load(n_rows=None):
    d = find_data()
    tr = pd.read_csv(d / "train.csv", nrows=n_rows)
    te = pd.read_csv(d / "test.csv", nrows=n_rows)
    ss = pd.read_csv(d / "sample_submission.csv", nrows=n_rows)
    bin_cols = [c for c in tr.columns if c.startswith("Target_Binary_")]
    prr_cols = [c for c in tr.columns if c.startswith("Target_PRR_")]
    effects = [c.removeprefix("Target_Binary_") for c in bin_cols]
    assert [c.removeprefix("Target_PRR_") for c in prr_cols] == effects, "binary/PRR column order differs"
    assert list(ss.columns) == ["Pair_ID", "Severity"] + bin_cols + prr_cols, "submission schema changed"
    assert set(tr.Severity.unique()) <= set(SEV_CLASSES), f"new severity class: {tr.Severity.unique()}"
    log(f"loaded train {tr.shape} test {te.shape}; {len(effects)} effects")
    return tr, te, ss, bin_cols, prr_cols, effects


def import_rdkit(allow_install=True):
    for attempt in (0, 1):
        try:
            from rdkit import Chem, RDLogger
            RDLogger.DisableLog("rdApp.*")
            return Chem
        except ImportError:
            if attempt or not allow_install:
                return None
            try:
                import subprocess
                import sys
                log("rdkit not installed: trying pip install (needs Internet on)")
                subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rdkit"],
                               check=True, timeout=600)
                log("pip install rdkit finished")
            except Exception as e:
                log(f"rdkit install failed: {repr(e)[:160]}")
                return None
    return None


def canonical_smiles(series_list):
    allsmi = pd.unique(pd.concat(series_list).dropna())
    Chem = import_rdkit()
    if Chem is None:
        log("!" * 100)
        log("WARNING: rdkit NOT available. Dropping ~180 chemistry features (Morgan fingerprints and")
        log("         physicochemical descriptors) and using raw SMILES as drug keys. Raw keys are")
        log("         still consistent within the run, so the target encodings are unaffected, but")
        log("         this run is NOT the intended V1. Enable Internet on the notebook, or add the")
        log("         rdkit package, and re-run to get the chemistry features.")
        log("!" * 100)
        return {s: s for s in allsmi}, False
    out, bad = {}, 0
    for s in allsmi:
        m = Chem.MolFromSmiles(s)
        if m is None:
            bad += 1
            out[s] = s
        else:
            out[s] = Chem.MolToSmiles(m)
    log(f"canonicalised {len(allsmi)} SMILES ({bad} unparseable), {len(set(out.values()))} distinct drugs")
    return out, True


def add_keys(df, canon):
    df = df.copy()
    for s in "AB":
        df[f"k{s}"] = df[f"SMILES_{s}"].map(lambda x: canon.get(x, x))
    df["pair_key"] = ["||".join(sorted(p)) for p in zip(df.kA, df.kB)]
    for s in "AB":
        df[f"txt{s}"] = df[[f"{f}_{s}" for f in FIELDS]].astype(str).agg(" ".join, axis=1).str.lower()
    return df


def parse_num(s, pat):
    m = re.search(pat, str(s))
    return float(m.group(1)) if m else np.nan


def cyp_set(s):
    return frozenset(t.upper() for t in CYP_RE.findall(str(s)))


def row_position_features(n, bp):
    i = np.arange(n, dtype=float)
    f = {"pos_idx": i, "pos_frac": i / max(n - 1, 1)}
    b1, b2 = bp
    f["pos_to_bp1"] = (i - b1) / n
    f["pos_to_bp2"] = (i - b2) / n
    f["pos_blk_major"] = (i < b1).astype(float)
    f["pos_blk_minor"] = (i >= b2).astype(float)
    f["pos_bnd_dist"] = np.minimum(np.abs(i - b1), np.abs(i - b2)) / n
    return f


def static_features(df, bp):
    f = dict(row_position_features(len(df), bp))
    ca = df.CYP450_Enzymes_A.map(cyp_set)
    cb = df.CYP450_Enzymes_B.map(cyp_set)
    f["cyp_n_a"] = ca.map(len).values.astype(float)
    f["cyp_n_b"] = cb.map(len).values.astype(float)
    f["cyp_shared"] = np.array([len(a & b) for a, b in zip(ca, cb)], float)
    f["cyp_union"] = np.array([len(a | b) for a, b in zip(ca, cb)], float)
    f["cyp_jaccard"] = np.array([len(a & b) / len(a | b) if (a | b) else 0.0 for a, b in zip(ca, cb)])
    f["cyp_missing"] = (df.CYP450_Enzymes_A.isna() | df.CYP450_Enzymes_B.isna()).values.astype(float)
    for tok in ["3A4", "2D6", "1A2", "2C19", "2C9", "2C8", "3A5"]:
        ia = ca.map(lambda s, t=tok: t in s).values
        ib = cb.map(lambda s, t=tok: t in s).values
        f[f"cyp_{tok}_both"] = (ia & ib).astype(float)
        f[f"cyp_{tok}_any"] = (ia | ib).astype(float)
    kw_keys = []
    for k in KW:
        kl = k.lower()
        a = df.txtA.str.contains(kl, regex=False).values
        b = df.txtB.str.contains(kl, regex=False).values
        key = re.sub(r"\W+", "_", k)
        kw_keys.append(key)
        f[f"kw_{key}_any"] = (a | b).astype(float)
        f[f"kw_{key}_both"] = (a & b).astype(float)
    f["kw_n_any"] = np.sum([f[f"kw_{key}_any"] for key in kw_keys], 0)
    f["kw_n_both"] = np.sum([f[f"kw_{key}_both"] for key in kw_keys], 0)
    for s in "AB":
        f[f"pb_{s}"] = df[f"Protein_Binding_{s}"].map(
            lambda x: parse_num(x, r"(\d+(?:\.\d+)?)\s*%")).values
        f[f"hl_{s}"] = df[f"Half_Life_{s}"].map(
            lambda x: parse_num(x, r"(\d+(?:\.\d+)?)\s*(?:-\s*\d+(?:\.\d+)?\s*)?(?:hours|hrs|h\b)")).values
        f[f"smi_len_{s}"] = df[f"SMILES_{s}"].str.len().values.astype(float)
        f[f"txt_len_{s}"] = df[f"txt{s}"].str.len().values.astype(float)
        for fl in ["Indication", "Warning", "CYP450_Enzymes"]:
            f[f"na_{fl}_{s}"] = df[f"{fl}_{s}"].isna().values.astype(float)
    f["pb_min"] = np.fmin(f["pb_A"], f["pb_B"])
    f["pb_max"] = np.fmax(f["pb_A"], f["pb_B"])
    f["hl_min"] = np.fmin(f["hl_A"], f["hl_B"])
    f["hl_max"] = np.fmax(f["hl_A"], f["hl_B"])
    f["hl_ratio"] = np.log1p(f["hl_max"]) - np.log1p(f["hl_min"])
    f["same_drug"] = (df.kA.values == df.kB.values).astype(float)
    return f


def rdkit_features(drugs, have_rdkit, n_bits=1024, n_svd=48, seed=0):
    if not have_rdkit:
        return None, None
    try:
        from rdkit import Chem
        from rdkit.Chem import Crippen, Descriptors, rdFingerprintGenerator, rdMolDescriptors
        from sklearn.decomposition import TruncatedSVD
    except ImportError as e:
        log("rdkit/sklearn feature import failed:", repr(e)[:150])
        return None, None
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
    desc_fns = [("mw", Descriptors.MolWt), ("logp", Crippen.MolLogP), ("tpsa", rdMolDescriptors.CalcTPSA),
                ("hba", rdMolDescriptors.CalcNumHBA), ("hbd", rdMolDescriptors.CalcNumHBD),
                ("rotb", rdMolDescriptors.CalcNumRotatableBonds), ("rings", rdMolDescriptors.CalcNumRings),
                ("arom", rdMolDescriptors.CalcNumAromaticRings), ("heavy", Descriptors.HeavyAtomCount),
                ("frac_csp3", rdMolDescriptors.CalcFractionCSP3), ("heteroatoms", rdMolDescriptors.CalcNumHeteroatoms),
                ("stereo", rdMolDescriptors.CalcNumAtomStereoCenters)]
    fps, desc, ok = [], [], []
    for s in drugs:
        m = Chem.MolFromSmiles(s)
        if m is None:
            fps.append(np.zeros(n_bits, np.int8))
            desc.append([np.nan] * len(desc_fns))
            ok.append(False)
            continue
        fps.append(np.asarray(gen.GetFingerprintAsNumPy(m), dtype=np.int8))
        row = []
        for _, fn in desc_fns:
            try:
                row.append(float(fn(m)))
            except Exception:
                row.append(np.nan)
        desc.append(row)
        ok.append(True)
    fps = np.asarray(fps, dtype=np.float32)
    k = int(min(n_svd, min(fps.shape) - 1))
    emb = TruncatedSVD(n_components=k, random_state=seed).fit_transform(fps) if k >= 2 else fps
    D = pd.DataFrame(desc, index=list(drugs), columns=[f"desc_{n}" for n, _ in desc_fns])
    E = pd.DataFrame(emb, index=list(drugs), columns=[f"fp{j}" for j in range(emb.shape[1])])
    log(f"rdkit: {len(drugs)} drugs, {sum(ok)} parsed, fp->{emb.shape[1]} svd dims, {D.shape[1]} descriptors")
    return E, D


def text_svd(tr, te, n_svd=64, max_features=60000, seed=0):
    try:
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError as e:
        log("sklearn text import failed:", repr(e)[:150])
        return None
    drug_txt = {}
    for df in (tr, te):
        for s in "AB":
            for k, t in zip(df[f"k{s}"].values, df[f"txt{s}"].values):
                drug_txt.setdefault(k, t)
    keys = list(drug_txt)
    vec = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=3, sublinear_tf=True,
                          strip_accents="unicode")
    X = vec.fit_transform([drug_txt[k] for k in keys])
    k_sv = int(min(n_svd, min(X.shape) - 1))
    if k_sv < 2:
        return None
    Z = TruncatedSVD(n_components=k_sv, random_state=seed).fit_transform(X)
    log(f"text: {len(keys)} drugs, tfidf {X.shape} -> svd {Z.shape}")
    return pd.DataFrame(Z, index=keys, columns=[f"tx{j}" for j in range(Z.shape[1])])


def pair_block(df, table, prefix):
    if table is None:
        return {}
    A = table.reindex(df.kA.values).to_numpy(dtype=np.float32)
    B = table.reindex(df.kB.values).to_numpy(dtype=np.float32)
    out = {}
    for j, c in enumerate(table.columns):
        a, b = A[:, j], B[:, j]
        out[f"{prefix}_{c}_sum"] = a + b
        out[f"{prefix}_{c}_absdiff"] = np.abs(a - b)
        out[f"{prefix}_{c}_prod"] = a * b
    return out


def drug_degree(tr, te):
    deg = pd.concat([tr.kA, tr.kB, te.kA, te.kB]).value_counts()
    deg_tr = pd.concat([tr.kA, tr.kB]).value_counts()
    return deg, deg_tr


def degree_features(df, deg, deg_tr):
    a = df.kA.map(deg).fillna(0).values.astype(float)
    b = df.kB.map(deg).fillna(0).values.astype(float)
    at = df.kA.map(deg_tr).fillna(0).values.astype(float)
    bt = df.kB.map(deg_tr).fillna(0).values.astype(float)
    return {"deg_min": np.minimum(a, b), "deg_max": np.maximum(a, b), "deg_sum": a + b,
            "degtr_min": np.minimum(at, bt), "degtr_max": np.maximum(at, bt),
            "degtr_zero": ((at == 0) | (bt == 0)).astype(float)}


def _loo_sums(codes, vals, n_drugs, row_idx):
    S = np.zeros((n_drugs, vals.shape[1]), dtype=np.float64)
    np.add.at(S, codes, vals[row_idx])
    C = np.bincount(codes, minlength=n_drugs).astype(np.float64)
    return S, C


def fold_target_encodings(tr, idx_fit, idx_apply_list, Y, R, sev, m=5.0):
    kA = tr.kA.values
    kB = tr.kB.values
    uniq = pd.unique(np.concatenate([kA, kB]))
    nd = len(uniq)
    kmap = {k: i for i, k in enumerate(uniq)}
    iA = pd.Series(kA).map(kmap).to_numpy(dtype=np.int64)
    iB = pd.Series(kB).map(kmap).to_numpy(dtype=np.int64)

    fit_mask = np.zeros(len(tr), bool)
    fit_mask[idx_fit] = True
    long_rows = np.concatenate([np.arange(len(tr)), np.arange(len(tr))])
    long_codes = np.concatenate([iA, iB])
    keep = fit_mask[long_rows]
    lc, lr = long_codes[keep], long_rows[keep]

    act = (Y.sum(1) > 0).astype(np.float64)[:, None]
    sev_oh = (sev[:, None] == np.array(SEV_CLASSES)[None, :]).astype(np.float64)
    prr_pos = (R > 0).astype(np.float64)
    blocks = [("sev", sev_oh, sev_oh[idx_fit].mean(0)),
              ("act", act, act[idx_fit].mean(0)),
              ("lab", Y.astype(np.float64), Y[idx_fit].mean(0))]
    sums = {}
    for name, V, prior in blocks:
        S = np.zeros((nd, V.shape[1]))
        np.add.at(S, lc, V[lr])
        sums[name] = (S, prior, V)
    Sp = np.zeros((nd, R.shape[1]))
    Cp = np.zeros((nd, R.shape[1]))
    np.add.at(Sp, lc, R[lr])
    np.add.at(Cp, lc, prr_pos[lr])
    prr_prior = np.where(prr_pos[idx_fit].sum(0) > 0,
                         R[idx_fit].sum(0) / np.maximum(prr_pos[idx_fit].sum(0), 1),
                         R[idx_fit][R[idx_fit] > 0].mean() if (R[idx_fit] > 0).any() else 1.0)
    C_rows = np.bincount(lc, minlength=nd).astype(np.float64)

    outs = []
    for idx in idx_apply_list:
        is_fit_row = fit_mask[idx]
        same = (iA[idx] == iB[idx]).astype(np.float64)[:, None]
        self_w = is_fit_row.astype(np.float64)[:, None] * (1.0 + same)
        f = {}
        for name, (S, prior, V) in sums.items():
            enc = []
            for side in (iA[idx], iB[idx]):
                n = C_rows[side][:, None] - self_w
                s = S[side] - V[idx] * self_w
                enc.append((s + m * prior) / np.maximum(n + m, 1e-9))
            a, b = enc
            if name == "lab":
                f["te_lab_mean"] = (a + b) / 2
                f["te_lab_min"] = np.minimum(a, b)
                f["te_lab_prod"] = a * b
            elif name == "sev":
                f["te_sev_mean"] = (a + b) / 2
                f["te_sev_prod"] = a * b / np.maximum(prior, 1e-9)
            else:
                f["te_act_mean"] = (a + b) / 2
                f["te_act_min"] = np.minimum(a, b)
        enc = []
        for side in (iA[idx], iB[idx]):
            n = Cp[side] - prr_pos[idx] * self_w
            s = Sp[side] - R[idx] * self_w
            enc.append((s + m * prr_prior) / np.maximum(n + m, 1e-9))
        f["te_prr_mean"] = (enc[0] + enc[1]) / 2
        f["te_prr_add"] = np.maximum(enc[0] + enc[1] - prr_prior, 0.01)
        outs.append(f)
    return outs, prr_prior


def flatten(feature_dicts, n):
    cols, names = [], []
    for d in feature_dicts:
        for k in sorted(d):
            v = np.asarray(d[k], dtype=np.float32)
            if v.ndim == 1:
                assert len(v) == n, f"{k}: length {len(v)} != {n}"
                cols.append(v[:, None])
                names.append(k)
            else:
                assert v.shape[0] == n, f"{k}: rows {v.shape[0]} != {n}"
                cols.append(v)
                names.extend([f"{k}_{j}" for j in range(v.shape[1])])
    X = np.concatenate(cols, 1) if cols else np.zeros((n, 0), np.float32)
    assert X.shape[1] == len(names), "feature/name count mismatch"
    return X, names


def build_static(tr, te, n_svd_fp=48, n_svd_tx=64, seed=0):
    canon, have_rdkit = canonical_smiles([tr.SMILES_A, tr.SMILES_B, te.SMILES_A, te.SMILES_B])
    tr = add_keys(tr, canon)
    te = add_keys(te, canon)
    drugs = sorted(set(tr.kA) | set(tr.kB) | set(te.kA) | set(te.kB))
    E, D = rdkit_features(drugs, have_rdkit, n_svd=n_svd_fp, seed=seed)
    TX = text_svd(tr, te, n_svd=n_svd_tx, seed=seed)
    deg, deg_tr = drug_degree(tr, te)
    out = []
    for df, bp in ((tr, TRAIN_BP), (te, TEST_BP)):
        d = [static_features(df, bp), degree_features(df, deg, deg_tr),
             pair_block(df, E, "fp"), pair_block(df, D, "desc"), pair_block(df, TX, "tx")]
        out.append({k: v for dd in d for k, v in dd.items()})
    log(f"static features: {len(out[0])} groups/columns for train, {len(out[1])} for test")
    return tr, te, out[0], out[1]
