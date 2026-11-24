import math
import time

import cv2
import numpy as np
import pandas as pd

from common.runtime import find_folder, run_session
from cv.config import Config
from cv.rotation import NccMatcher, SiftMatcher, angle_error, synthetic_set, to_label


def has_inputs(folder):
    return (folder / "images" / "base.jpg").exists() and \
        {"id", "img"} <= set(pd.read_csv(folder / "test.csv", nrows=1).columns)


def read_image(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    assert img is not None, path
    return img


def estimate_all(images, ncc, sift):
    rows = []
    for img in images:
        a, score = ncc.estimate(img)
        s = sift.estimate(img)
        rows.append({"ncc_angle": a, "ncc_score": score, "sift_angle": s["angle"], "sift_inliers": s["inliers"],
                     "sift_scale": s["scale"], "sift_shift": s["shift"]})
    df = pd.DataFrame(rows)
    df["sift_angle_filled"] = df["sift_angle"].where(df["sift_angle"].notna(), df["ncc_angle"])
    df["disagree"] = [angle_error(a, b) if not math.isnan(b) else math.nan
                      for a, b in zip(df["ncc_angle"], df["sift_angle"])]
    return df


def labels_of(angles, cfg):
    pairs = [to_label(a, cfg.label_max) for a in angles]
    return np.array([p[0] for p in pairs]), int(sum(p[1] for p in pairs))


def describe(name, df, cfg):
    print(f"{name}: ncc score min {df['ncc_score'].min():.3f} median {df['ncc_score'].median():.3f}; "
          f"sift failed {int(df['sift_angle'].isna().sum())}, inliers median {df['sift_inliers'].median():.0f}, "
          f"scale median {df['sift_scale'].median():.4f}, centre shift median {df['sift_shift'].median():.2f}px; "
          f"ncc vs sift > {cfg.disagree_deg} deg: {int((df['disagree'] > cfg.disagree_deg).sum())}, "
          f"median {df['disagree'].median():.3f}")


def write_submission(path, ids, labels, cfg):
    sub = pd.DataFrame({"id": ids, "label": labels})
    assert len(sub) == cfg.n_test and sub["id"].is_unique and np.isfinite(sub["label"]).all()
    assert sub["label"].between(0, cfg.label_max).all()
    sub.to_csv(path, index=False, float_format="%.4f")
    back = pd.read_csv(path, dtype={"id": str})
    assert list(back.columns) == ["id", "label"] and back["id"].tolist() == list(ids)


def solve(cfg, work, submission):
    folder = find_folder(cfg, "test.csv", has_inputs)
    test = pd.read_csv(folder / "test.csv", dtype=str)
    sample = pd.read_csv(folder / "sample_submission.csv", dtype=str)
    assert len(test) == cfg.n_test and test["id"].tolist() == sample["id"].tolist()
    base = read_image(folder / "images" / "base.jpg")
    print(f"data {folder}: {len(test)} test images, base {base.shape}")
    ncc, sift = NccMatcher(base, cfg), SiftMatcher(base, cfg)

    t0 = time.time()
    synth = synthetic_set(base, cfg)
    sdf = estimate_all([img for _, img in synth], ncc, sift)
    sdf["theta"] = [t for t, _ in synth]
    mse = {}
    for arm in ["ncc_angle", "sift_angle_filled"]:
        pred, _ = labels_of(sdf[arm], cfg)
        sdf[f"{arm}_label"] = pred
        mse[arm] = float(np.mean((pred - sdf["theta"]) ** 2))
    sdf.to_csv(work / "extra" / "synthetic.csv", index=False)
    describe("synthetic", sdf, cfg)
    print(f"synthetic MSE {mse} ({time.time() - t0:.0f}s)")

    t0 = time.time()
    images = [read_image(folder / p) for p in test["img"]]
    tdf = estimate_all(images, ncc, sift)
    tdf.insert(0, "id", test["id"].values)
    describe("test", tdf, cfg)
    print(f"test estimated in {time.time() - t0:.0f}s")
    arms, out_of_range = {}, {}
    for arm in ["ncc_angle", "sift_angle_filled"]:
        arms[arm], out_of_range[arm] = labels_of(tdf[arm], cfg)
        tdf[f"{arm}_label"] = arms[arm]
        write_submission(work / "extra" / f"submission_{arm}.csv", test["id"].values, arms[arm], cfg)
    tdf.to_csv(work / "extra" / "test_angles.csv", index=False)
    print(f"raw angles outside [-5, {cfg.label_max + 5:.0f}] (direction check): {out_of_range}")
    worst = tdf.sort_values("disagree", ascending=False).head(15)
    print("largest ncc/sift disagreements\n" + worst.to_string(index=False))

    pick = min(mse, key=mse.get)
    labels = arms[pick]
    print(f"pick {pick}; label mean {labels.mean():.2f}, share at 0 {np.mean(labels == 0):.3f}, "
          f"at {cfg.label_max:.0f} {np.mean(labels == cfg.label_max):.3f}")
    write_submission(submission, test["id"].values, labels, cfg)
    return {"synthetic_mse": mse, "pick": pick, "out_of_range": out_of_range,
            "test_disagree_median": float(tdf["disagree"].median())}


def run(cfg=None):
    run_session(cfg or Config(), solve)


if __name__ == "__main__":
    run()
