from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "exam_score"
NUMERIC = ["study_hours", "sleep_hours", "attendance_pct", "previous_score", "screen_time_hours"]
ORDINAL = {
    "tutoring": {"none": 0, "weekly": 1, "daily": 2},
    "parent_education": {"school": 0, "college": 1, "graduate": 2, "postgraduate": 3},
}
BINARY = {
    "internet_access": {"no": 0, "yes": 1},
    "extracurricular": {"no": 0, "yes": 1},
    "school_type": {"public": 0, "private": 1},
}
FEATURES = NUMERIC + list(ORDINAL) + list(BINARY)


def get_data_dir() -> Path:
    kaggle = Path("/kaggle/input")
    if kaggle.exists():
        for p in sorted(kaggle.rglob("train.csv")):
            with open(p, encoding="utf-8") as f:
                if "screen_time_hours" in f.readline():
                    return p.parent
        raise FileNotFoundError("no train.csv with screen_time_hours under /kaggle/input")
    try:
        return Path(__file__).resolve().parent.parent / "data"
    except NameError:
        return Path.cwd() / "data"


def get_out_dir() -> Path:
    return Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")


def encode(df):
    out = df[["id"] + NUMERIC].copy()
    for col, mapping in {**ORDINAL, **BINARY}.items():
        unknown = set(df[col].dropna().unique()) - set(mapping)
        assert not unknown, f"{col}: unexpected levels {unknown}"
        out[col] = df[col].map(mapping).astype("float64")
    return out


def load_data():
    data_dir = get_data_dir()
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    sample = pd.read_csv(data_dir / "sample_submission.csv")
    print(f"data dir: {data_dir}  train {train.shape}  test {test.shape}  sample {sample.shape}")

    assert len(train) == 7000 and len(test) == 3000, "unexpected row counts"
    assert train["id"].is_unique and test["id"].is_unique
    assert (test["id"].to_numpy() == sample["id"].to_numpy()).all(), "test id order differs from sample_submission"
    assert train[TARGET].notna().all()

    X = encode(train)[FEATURES]
    X_test = encode(test)[FEATURES]
    y = train[TARGET].to_numpy(dtype="float64")
    print("missing per feature (train / test):")
    for c in FEATURES:
        n_tr, n_te = int(X[c].isna().sum()), int(X_test[c].isna().sum())
        if n_tr or n_te:
            print(f"  {c:18s} {n_tr:4d} / {n_te:4d}")
    print(f"target mean {y.mean():.3f}  sd {y.std():.3f}")
    return X, y, X_test, train["id"].to_numpy(), sample


if __name__ == "__main__":
    X, y, X_test, train_ids, sample = load_data()
    print(X.describe().T.round(2))
    print(np.isfinite(y).all())
