import ast
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

LABELS_CSV, JUDGE_CSV, SAMPLE_CSV = "wiki_labels.csv", "wiki_judge.csv", "sample_submission.csv"
IMAGE_DIRS = {"train": "wiki_labeled/wiki_labeled", "judge": "wiki_judge_images/wiki_judge_images"}
CACHE = {"train": ("train_images.npy", "train_ids.npy"), "judge": ("judge_images.npy", "judge_ids.npy")}
SIZE = 100
AGE_MAX = 100
NO_FACE = "#NAME?"


def local_data() -> Path:
    try:
        return Path(__file__).resolve().parent.parent / "data"
    except NameError:
        return Path.cwd() / "data"


def find(name):
    root = Path("/kaggle/input")
    if not root.exists():
        p = local_data() / name
        return p if p.exists() else None
    for d, dirs, files in os.walk(root):
        dirs[:] = [x for x in dirs if x not in ("wiki_labeled", "wiki_judge_images")]
        if name in files:
            return Path(d) / name
    return None


def load_tables():
    d = find(LABELS_CSV).parent
    lab, judge, sample = (pd.read_csv(d / f) for f in (LABELS_CSV, JUDGE_CSV, SAMPLE_CSV))
    assert list(lab.columns) == ["ID", "dob", "dob_str", "photo_taken", "full_path", "gender", "name",
                                 "face_location", "face_score", "second_face_score", "age"]
    assert list(judge.columns) == ["ID", "full_path", "gender", "face_score", "second_face_score"]
    assert list(sample.columns) == ["ID", "age"] and sample.ID.is_unique and judge.ID.isin(sample.ID).all()
    assert lab.ID.is_unique and not lab.ID.isin(sample.ID).any()
    for df in (lab, judge):
        df["path"] = df.full_path.map(lambda s: ast.literal_eval(s)[0])
    assert (judge.path == judge.ID.astype(str) + ".png").all()
    lab["face"] = lab.face_score.ne(NO_FACE)
    assert pd.to_numeric(lab.face_score[lab.face], errors="coerce").notna().all()
    lab["clean"] = lab.face & (lab.age > 0) & (lab.age <= AGE_MAX)
    print(f"labeled {len(lab)} (face {lab.face.sum()}, clean {lab.clean.sum()}), judge {len(judge)}, "
          f"sample ids {len(sample)} from {d}")
    return d, lab, judge, sample


def _read(path):
    with Image.open(path) as im:
        a = np.asarray(im)
    if a.dtype == np.uint16:
        a = (a >> 8).astype(np.uint8)
    assert a.shape == (SIZE, SIZE) and a.dtype == np.uint8, (path, a.shape, a.dtype)
    return a


def decode(paths, threads=32):
    with ThreadPoolExecutor(threads) as ex:
        return np.stack(list(ex.map(_read, paths)))


def load_images(d, part, df, rows):
    folder = d / IMAGE_DIRS[part]
    images_file, ids_file = CACHE[part]
    p = find(images_file)
    if p is None:
        print(f"{part}: no age-cache dataset, decoding {len(rows)} PNGs")
        return decode([folder / s for s in df.path.to_numpy()[rows]])
    assert (np.load(p.parent / ids_file) == df.ID.to_numpy()).all(), f"{p} rows are not {part} csv rows"
    images = np.load(p, mmap_mode="r")[rows]
    check = np.random.default_rng(0).choice(len(rows), min(16, len(rows)), replace=False)
    assert (decode([folder / df.path.iloc[rows[i]] for i in check]) == images[check]).all(), f"{p} != PNGs"
    return images
