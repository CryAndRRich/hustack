import importlib.util
import math
import pickle
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image

from detector import degrade
from fusion import FLIP, arms as make_arms, fuse


def load_scorer(kit):
    spec = importlib.util.spec_from_file_location("kit_score", kit / "score.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def speed_score(ms, fast_ms, slow_ms):
    if ms <= fast_ms:
        return 1.0
    if ms >= slow_ms:
        return 0.0
    return (math.log(slow_ms) - math.log(ms)) / (math.log(slow_ms) - math.log(fast_ms))


def read_rgb(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def accuracy(scorer, gt, preds, ids):
    m = scorer.evaluate(gt, preds, ids)
    return {"accuracy": scorer.combine(m)["accuracy_score"], "mAP50": m["mAP50"], "mAP50_95": m["mAP50_95"],
            "precision": m["precision"], "recall": m["recall"], "threshold": m["confidence_threshold"],
            "per_class": m["per_class"]}


def collect(models, folder, ids, degraded=False):
    store = {m + s: {} for m in models for s in ("", FLIP)}
    sizes = {}
    t0 = time.time()
    for i, iid in enumerate(ids):
        image = read_rgb(folder / f"{iid}.jpg")
        if degraded:
            image = degrade(image, i)
        sizes[iid] = (image.shape[1], image.shape[0])
        for name, model in models.items():
            store[name][iid], store[name + FLIP][iid] = model(image, True)
    print(f"collect {folder.name}{' degraded' if degraded else ''}: {len(ids)} images, {len(models)} models "
          f"in {time.time() - t0:.0f}s")
    return store, sizes


def fused(store, arm, ids, cfg):
    return {iid: fuse([store[m][iid] for m in arm["members"]], cfg.wbf_iou, cfg.max_det) for iid in ids}


def predict_arm(models, arm, image, cfg):
    parts = [p for name in arm["models"] for p in models[name](image, arm["flip"])]
    return fuse(parts, cfg.wbf_iou, cfg.max_det)


def latency_ms(models, arm, images, cfg):
    for im in images[:cfg.warmup]:
        predict_arm(models, arm, im, cfg)
    times = []
    for im in images:
        torch.cuda.synchronize()
        t = time.perf_counter()
        predict_arm(models, arm, im, cfg)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t) * 1000)
    return float(np.median(times)), float(np.percentile(times, 95))


def format_rows(rows):
    return " ".join(f"{int(c)} {s:.5f} {x1:.2f} {y1:.2f} {x2:.2f} {y2:.2f}" for c, s, x1, y1, x2, y2 in rows) or "none"


def write_rows(path, ids, preds):
    pd.DataFrame({"image_id": ids, "PredictionString": [format_rows(preds.get(i, [])) for i in ids]}).to_csv(
        path, index=False)


def save_store(path, store, sizes):
    with open(path, "wb") as f:
        pickle.dump({"sizes": sizes, "members": {m: {k: v.astype(np.float32) for k, v in d.items()}
                                                 for m, d in store.items()}}, f)


def select(cfg, kit, models, extra):
    scorer = load_scorer(kit)
    ids = scorer.load_ids(str(kit / "val_image_ids.csv"))
    ids = ids[:cfg.eval_limit] if cfg.eval_limit else ids
    gt = scorer.load_gt(str(kit / "val_labels.csv"))
    store, sizes = collect(models, kit / "val_images", ids)
    save_store(extra / "members_val.pkl", store, sizes)
    widths = {iid: s[0] for iid, s in sizes.items()}
    arms = make_arms(list(models), cfg.max_subset)
    scores = {}
    t0 = time.time()
    for name, arm in arms.items():
        scores[name] = accuracy(scorer, gt, fused(store, arm, ids, cfg), ids)
        s = scores[name]
        print(f"arm {name}: accuracy {s['accuracy']:.3f} mAP50 {s['mAP50']:.4f} mAP50:95 {s['mAP50_95']:.4f} "
              f"P {s['precision']:.4f} R {s['recall']:.4f}")
    print(f"{len(arms)} arms scored in {time.time() - t0:.0f}s")
    ranked = sorted(arms, key=lambda a: -scores[a]["accuracy"])
    singles = [a for a in ranked if len(arms[a]["members"]) == 1]
    candidates = list(dict.fromkeys(ranked[:cfg.top_arms] + singles[:1]))
    deg_store, deg_sizes = collect(models, kit / "val_images", ids, degraded=True)
    lat_images = [read_rgb(kit / "val_images" / f"{iid}.jpg") for iid in ids[:cfg.latency_images]]
    report = {}
    for name in candidates:
        arm = arms[name]
        preds = fused(store, arm, ids, cfg)
        by_width = {}
        for w in sorted(set(widths.values())):
            sub = [i for i in ids if widths[i] == w]
            if sum(len(gt[i][1]) for i in sub if i in gt):
                by_width[int(w)] = accuracy(scorer, gt, preds, sub)["accuracy"]
        robustness = accuracy(scorer, gt, fused(deg_store, arm, ids, cfg), ids)["mAP50"]
        median_ms, p95_ms = latency_ms(models, arm, lat_images, cfg)
        speed = speed_score(median_ms, cfg.fast_ms, cfg.slow_ms)
        estimate = 0.85 * scores[name]["accuracy"] + 10 * speed + 5 * robustness
        report[name] = {"val": scores[name], "accuracy_by_width": by_width, "robustness_synthetic": robustness,
                        "median_ms": median_ms, "p95_ms": p95_ms, "speed_score": speed, "final_estimate": estimate}
        write_rows(extra / f"val_{name}.csv", ids, preds)
        reread = accuracy(scorer, gt, scorer.load_predictions(str(extra / f"val_{name}.csv")), ids)["accuracy"]
        assert abs(reread - scores[name]["accuracy"]) < 0.05, f"written val file scores {reread}"
        print(f"candidate {name}: accuracy {scores[name]['accuracy']:.3f} by width {by_width} robustness "
              f"{robustness:.4f} latency {median_ms:.1f} ms (p95 {p95_ms:.1f}) speed {speed:.3f} "
              f"estimate {estimate:.2f}")
    chosen = max(report, key=lambda a: report[a]["final_estimate"])
    print(f"chosen arm: {chosen} (most accurate: {ranked[0]})")
    return chosen, report, {a: scores[a]["accuracy"] for a in ranked}, arms


def predict_test(cfg, kit, models, arms, report, chosen, submission, extra):
    scorer = load_scorer(kit)
    ids = scorer.load_ids(str(kit / "test_image_ids.csv"))
    ids = ids[:cfg.eval_limit] if cfg.eval_limit else ids
    store, sizes = collect(models, kit / "test_images", ids)
    save_store(extra / "members_test.pkl", store, sizes)
    paths = {a: submission if a == chosen else extra / f"submission_{a}.csv" for a in report}
    for name, path in paths.items():
        write_rows(path, ids, fused(store, arms[name], ids, cfg))
        df = pd.read_csv(path, keep_default_na=False)
        assert list(df.columns) == ["image_id", "PredictionString"] and df.image_id.tolist() == ids
        assert (df.PredictionString.str.len() > 0).all()
        parsed = scorer.load_predictions(str(path), valid_ids=set(ids))
        n_boxes = sum(len(v) for v in parsed.values())
        print(f"{name}: wrote {path} with {len(df)} rows, {n_boxes} boxes, "
              f"{int((df.PredictionString == 'none').sum())} none")
    return {a: str(p) for a, p in paths.items()}
