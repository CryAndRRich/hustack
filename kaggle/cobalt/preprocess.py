import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

T0 = time.time()
OUT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
ROCKS = [1, 2, 3, 4, 5]
LANDUSES = [1, 2, 3, 4]
SITE = ["Xloc", "Yloc", "Landuse", "Rock"]
METALS = ["Cd", "Co", "Cr", "Cu", "Ni", "Pb", "Zn"]
SITES_FILE = "cobalt_sites_v3.csv"
UNIT, LITHO = "geologie-geocover:description_de", "geologie-geocover:litho_de"
SHARES_DISTS = ([f"rock{r}_share_{m}m" for m in (100, 250, 500) for r in range(1, 6)]
                + [f"dist_rock{r}" for r in range(1, 6)])
AEF_PCS = 8


def log(msg):
    print(f"[{(time.time() - T0) / 60:6.2f} min] {msg}", flush=True)


def find_dir(fname, local_dir):
    kaggle = Path("/kaggle/input")
    if kaggle.exists():
        hits = sorted(kaggle.rglob(fname))
        if hits:
            return hits[0].parent
    try:
        local = Path(__file__).resolve().parent.parent / local_dir
    except NameError:
        local = Path.cwd() / local_dir
    assert (local / fname).exists(), f"{fname} not found under /kaggle/input or {local}"
    return local


def load():
    d = find_dir("jura_pred.csv", "data")
    train = pd.read_csv(d / "jura_pred.csv")
    test = pd.read_csv(d / "jura_val_loc.csv").rename(columns={"seq(1:100)": "id"})
    assert len(train) == 259 and len(test) == 100, (len(train), len(test))
    assert list(test.columns) == ["id"] + SITE and (test["id"] == np.arange(1, 101)).all()
    for df in (train, test):
        assert df[SITE].notna().all().all()
        assert set(df["Rock"]) <= set(ROCKS) and set(df["Landuse"]) <= set(LANDUSES)
    assert train["Co"].notna().all() and (train["Co"] > 0).all()
    return train, test


def nearest(a, b, same=False):
    d = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))
    if same:
        np.fill_diagonal(d, np.inf)
    return d.min(1)


def describe(train, test):
    co = train["Co"]
    log(f"Co: mean {co.mean():.3f}, var {co.var(ddof=0):.3f}, quantiles 0/10/25/50/75/90/100 % "
        f"{np.round(co.quantile([0, .1, .25, .5, .75, .9, 1]).to_numpy(), 2).tolist()}")
    for f in ("Rock", "Landuse"):
        g = train.groupby(f)["Co"].agg(["count", "mean", "std"]).round(2)
        g["test_count"] = test[f].value_counts().reindex(g.index).fillna(0).astype(int)
        assert set(test[f]) <= set(g.index), f"test {f} level missing from train"
        log(f"Co by {f}:\n{g}")


def load_sites(train, test):
    s = pd.read_csv(find_dir(SITES_FILE, "data/sites") / SITES_FILE)
    assert s.loc[s["split"] == "test", [m for m in METALS if m in s]].isna().all().all(), \
        f"{SITES_FILE} must hold no test metal values"
    if UNIT in s:
        s["unit"] = pd.factorize(s[UNIT].where(s[UNIT] != "-", s[LITHO]).fillna("none"))[0]
    else:
        log("no GeoCover unit column; unit = 0")
        s["unit"] = 0
    for split, df in (("train", train), ("test", test)):
        part = s[s["split"] == split].reset_index(drop=True)
        assert len(part) == len(df) and np.allclose(part[SITE].to_numpy(float), df[SITE].to_numpy(float)), \
            f"{SITES_FILE} {split} rows do not match the competition file"
    assert np.allclose(s.loc[s["split"] == "train", "Co"].to_numpy(), train["Co"].to_numpy())
    blocks = {"base": ["Rock", "Landuse", "unit"] + SHARES_DISTS + ["Xloc", "Yloc"],
              "terrain": [c for c in s if c.startswith("dem_")],
              "ext": [c for c in s if c.startswith("ext_")]}
    aef = [c for c in s if c.startswith("aef_")]
    if aef:
        a = s[aef].to_numpy(float)
        u, sv, _ = np.linalg.svd(a - a.mean(0), full_matrices=False)
        for k in range(AEF_PCS):
            s[f"aef_pc{k}"] = u[:, k] * sv[k]
        log(f"AEF: PCA of {len(aef)} dims, top {AEF_PCS} explain {(sv[:AEF_PCS] ** 2).sum() / (sv ** 2).sum():.3f}")
    blocks["aef"] = [f"aef_pc{k}" for k in range(AEF_PCS)] if aef else []
    for c in blocks["ext"]:
        if s[c].isna().any():
            log(f"{c}: {int(s[c].isna().sum())} missing values filled with the median")
            s[c] = s[c].fillna(s[c].median())
    cols = [c for b in blocks.values() for c in b]
    assert np.isfinite(s[cols].to_numpy(float)).all(), "non-finite site features"
    log("feature blocks: " + ", ".join(f"{k} {len(v)}" for k, v in blocks.items()))
    train_all = s[s["split"] != "test"].reset_index(drop=True)
    test_all = s[s["split"] == "test"].reset_index(drop=True).assign(id=test["id"].to_numpy())
    log(f"training rows: {train_all['split'].value_counts().to_dict()}")
    return train_all, test_all, blocks


def shift_report(train, test, blocks):
    rows = {}
    for c in [c for b in blocks.values() for c in b if c not in ("Rock", "Landuse", "unit")]:
        a, b = train[c], test[c]
        rows[c] = {"train_min": a.min(), "train_max": a.max(), "test_min": b.min(), "test_max": b.max(),
                   "test_outside": int(((b < a.min()) | (b > a.max())).sum()), "ks": ks_2samp(a, b).statistic,
                   "ks_p": ks_2samp(a, b).pvalue}
    t = pd.DataFrame(rows).T
    log(f"train vs test: features with test values outside the train range or KS p < 0.05:\n"
        f"{t[(t['test_outside'] > 0) | (t['ks_p'] < 0.05)].round(3).to_string()}")
