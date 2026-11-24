import time
from pathlib import Path

import numpy as np
import pandas as pd

T0 = time.time()
OUT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
ROCKS = [1, 2, 3, 4, 5]
LANDUSES = [1, 2, 3, 4]
SITE = ["Xloc", "Yloc", "Landuse", "Rock"]


def log(msg):
    print(f"[{(time.time() - T0) / 60:6.2f} min] {msg}", flush=True)


def find_data_dir():
    kaggle = Path("/kaggle/input")
    if kaggle.exists():
        hits = sorted(kaggle.rglob("jura_pred.csv"))
        if hits:
            return hits[0].parent
    try:
        local = Path(__file__).resolve().parent.parent / "data"
    except NameError:
        local = Path.cwd() / "data"
    assert (local / "jura_pred.csv").exists(), f"jura_pred.csv not found under /kaggle/input or {local}"
    return local


def load():
    d = find_data_dir()
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
    log(f"Co: mean {co.mean():.3f}, var {co.var(ddof=0):.3f}, skew {co.skew():.2f} (log {np.log(co).skew():.2f}), "
        f"quantiles 0/10/25/50/75/90/100 % {np.round(co.quantile([0, .1, .25, .5, .75, .9, 1]).to_numpy(), 2).tolist()}")
    for f in ("Rock", "Landuse"):
        g = train.groupby(f)["Co"].agg(["count", "mean", "std"]).round(2)
        g["test_count"] = test[f].value_counts().reindex(g.index).fillna(0).astype(int)
        assert set(test[f]) <= set(g.index), f"test {f} level missing from train"
        log(f"Co by {f}:\n{g}")
    log(f"Co mean by Rock (rows) x Landuse (columns):\n"
        f"{train.pivot_table(index='Rock', columns='Landuse', values='Co', aggfunc='mean').round(2)}")
    xy_tr, xy_te = train[["Xloc", "Yloc"]].to_numpy(), test[["Xloc", "Yloc"]].to_numpy()
    q = [0, .25, .5, .75, 1]
    log(f"nearest train site (km), quantiles 0/25/50/75/100 %: from train "
        f"{np.round(np.quantile(nearest(xy_tr, xy_tr, True), q), 3).tolist()}, from test "
        f"{np.round(np.quantile(nearest(xy_te, xy_tr), q), 3).tolist()}")
    log(f"metal correlations with Co: {train.drop(columns=SITE).corr()['Co'].round(2).to_dict()}")
