import glob
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import GroupKFold

H, W, CELL = 720, 1280, 16
GH, GW = H // CELL, W // CELL
N_FOLDS = 5
DUP_CORR = 0.95


def find_data():
    for f in glob.glob("/kaggle/input/**/train.csv", recursive=True):
        if "density" in pd.read_csv(f, nrows=1).columns:
            return Path(f).parent
    try:
        return Path(__file__).resolve().parent.parent / "data"
    except NameError:
        return Path.cwd() / "data"


def load():
    root = find_data()
    tr, te = pd.read_csv(root / "train.csv"), pd.read_csv(root / "sample_submission.csv")
    paths = {f[:-4]: os.path.join(d, f) for d, _, fs in os.walk(root) for f in fs if f.endswith(".jpg")}
    ids = [*tr["id"], *te["id"]]
    missing = [i for i in ids if i not in paths]
    assert not missing, f"missing images under {root}: {missing[:5]}"
    with ThreadPoolExecutor(16) as ex:
        imgs = np.stack(list(ex.map(lambda i: np.asarray(Image.open(paths[i]).convert("RGB")), ids)))
    assert imgs.shape[1:] == (H, W, 3), imgs.shape
    Y = np.stack([np.array(s.split(), np.float32).reshape(GH, GW) for s in tr["density"]])
    assert len(tr) == 100 and len(te) == 200 and (Y >= 0).all()
    return tr, te, imgs, Y


def load_dense_chicken():
    found = [d for d, _, fs in os.walk("/kaggle/input") if "dc_maps.npz" in fs]
    assert found, "attach the dense-chicken dataset (dc_maps.npz + dc_<id>.jpg)"
    z = np.load(Path(found[0]) / "dc_maps.npz")
    with ThreadPoolExecutor(16) as ex:
        imgs = np.stack(list(ex.map(lambda i: np.asarray(Image.open(Path(found[0]) / f"dc_{i}.jpg").convert("RGB")),
                                    z["ids"])))
    assert imgs.shape == (len(z["ids"]), H, W, 3) and z["maps"].shape[1:] == (H // 5, W // 5), (imgs.shape, z["maps"].shape)
    return imgs, z["maps"]


def make_folds(imgs):
    t = np.stack([im.astype(np.float32).mean(-1).reshape(36, 20, 64, 20).mean((1, 3)).ravel() for im in imgs])
    t = (t - t.mean(1, keepdims=True)) / t.std(1, keepdims=True)
    corr = t @ t.T / t.shape[1]
    parent = list(range(len(imgs)))

    def root(i):
        while parent[i] != i:
            i = parent[i]
        return i

    for i, j in zip(*np.nonzero(np.triu(corr, 1) > DUP_CORR)):
        parent[root(i)] = root(j)
    groups = np.array([root(i) for i in range(len(imgs))])
    sizes = np.bincount(np.unique(groups, return_inverse=True)[1])
    print(f"folds: {len(sizes)} groups of near-duplicate frames (corr > {DUP_CORR}), largest {sizes.max()}")
    fold = np.zeros(len(imgs), np.int64)
    for k, (_, b) in enumerate(GroupKFold(N_FOLDS).split(imgs, groups=groups)):
        fold[b] = k
    return fold
