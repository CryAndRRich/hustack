from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide",
            "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"]
TARGET, ID = "quality", "id"
LABELS = np.arange(3, 9)
EXT_FILES = {"uci": "winequality-red.csv", "s3e5": "s3e5_train.csv"}


def local_data() -> Path:
    try:
        return Path(__file__).resolve().parent.parent / "data"
    except NameError:
        return Path.cwd() / "data"


def competition_dir() -> Path:
    kaggle = Path("/kaggle/input")
    if not kaggle.exists():
        return local_data()
    for p in sorted(kaggle.rglob("train.csv")):
        with open(p, encoding="utf-8") as f:
            head = f.readline()
        if head.startswith(f"{ID},") and "volatile acidity" in head and TARGET in head:
            return p.parent
    raise FileNotFoundError("no competition train.csv (id, ..., quality) under /kaggle/input")


def load_external() -> pd.DataFrame | None:
    root = Path("/kaggle/input") if Path("/kaggle/input").exists() else local_data() / "external"
    frames = []
    for source, name in EXT_FILES.items():
        hits = sorted(root.rglob(name))
        if not hits:
            continue
        df = pd.read_csv(hits[0])
        missing = set(FEATURES + [TARGET]) - set(df.columns)
        assert not missing, f"{name}: missing {missing}; re-upload wine-external.zip"
        frames.append(df[FEATURES + [TARGET]].assign(source=source))
    if not frames:
        print("wine-external not attached: the *_ext arms are skipped")
        return None
    ext = pd.concat(frames, ignore_index=True)
    assert ext[TARGET].isin(LABELS).all()
    print("external rows:", ext["source"].value_counts().to_dict())
    return ext


def load():
    d = competition_dir()
    train, test = pd.read_csv(d / "train.csv"), pd.read_csv(d / "test.csv")
    sample = pd.read_csv(d / "sample_submission.csv")
    assert list(train.columns) == [ID, *FEATURES, TARGET] and list(test.columns) == [ID, *FEATURES]
    assert list(sample.columns) == [ID, TARGET] and (sample[ID].to_numpy() == test[ID].to_numpy()).all()
    assert (train[TARGET] % 1 == 0).all()
    train[TARGET] = train[TARGET].astype(int)
    assert train[TARGET].isin(LABELS).all()
    print(f"train {train.shape}, test {test.shape} from {d}")
    return train, test, sample, load_external()
