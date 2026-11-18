import pickle
from pathlib import Path

import numpy as np

ALLOWED = {("numpy.core.multiarray", "_reconstruct"), ("numpy", "ndarray"), ("numpy", "dtype")}


class NumpyOnly(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) not in ALLOWED:
            raise pickle.UnpicklingError(f"blocked global {module}.{name}")
        return super().find_class(module, name)


def data_dir():
    for p in Path("/kaggle/input").rglob("cifar-100-python/train"):
        return p.parent
    try:
        return Path(__file__).resolve().parent.parent / "data" / "cifar-100-python"
    except NameError:
        return Path.cwd() / "data" / "cifar-100-python"


def read(path):
    with open(path, "rb") as f:
        d = NumpyOnly(f, encoding="bytes").load()
    x = d[b"data"].reshape(-1, 3, 32, 32)
    y = np.asarray(d[b"fine_labels"], dtype=np.int64)
    return x, y


def load():
    d = data_dir()
    xtr, ytr = read(d / "train")
    xte, yte = read(d / "test")
    assert xtr.shape == (50000, 3, 32, 32) and xte.shape == (10000, 3, 32, 32)
    assert (np.bincount(ytr) == 500).all() and (np.bincount(yte) == 100).all()
    print("data", d, xtr.shape, xte.shape)
    return xtr, ytr, xte, yte
