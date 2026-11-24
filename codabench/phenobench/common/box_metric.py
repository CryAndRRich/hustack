import json
import os
import subprocess
import sys

import numpy as np
import torch
from scipy import ndimage
from torchvision.ops import box_convert, box_iou

SIZE = 1024
IOU_T = torch.linspace(0.5, 0.95, 10)
REC_T = torch.linspace(0.0, 1.0, 101)
MAX_DET = 100
WRITE_MAX = 150
DUMMY = (1, (512, 512, 2, 2), 0.001)


def mask_boxes(m):
    objs = ndimage.find_objects(np.asarray(m, np.int32))
    boxes, ok = np.zeros((len(objs), 4), np.int64), np.zeros(len(objs), bool)
    for k, sl in enumerate(objs):
        if sl is not None:
            y0, y1, x0, x1 = sl[0].start, sl[0].stop - 1, sl[1].start, sl[1].stop - 1
            boxes[k] = round((x0 + x1) / 2), round((y0 + y1) / 2), x1 - x0, y1 - y0
            ok[k] = True
    return boxes, ok


def instances(inst, sem, vis):
    inst = np.asarray(inst)
    u, first, inv = np.unique(inst.ravel(), return_index=True, return_inverse=True)
    boxes = mask_boxes(inv.reshape(inst.shape) + int(u[0] != 0))[0]
    f = first[u > 0]
    label = np.ones(len(f), np.int64) if sem is None else np.asarray(sem).ravel()[f].astype(np.int64)
    return boxes, label, np.asarray(vis).ravel()[f]


def ground_truth(inst, sem, vis):
    boxes, label, v = instances(inst, sem, vis)
    keep, full = label > 0, v > 127
    return {"boxes": boxes[keep & full], "labels": label[keep & full], "partial": boxes[keep & ~full]}


def _rects(b):
    b = np.asarray(b, np.float64)
    r = np.stack([np.round(b[:, 0] - b[:, 2] / 2), np.round(b[:, 0] + b[:, 2] / 2),
                  np.round(b[:, 1] - b[:, 3] / 2), np.round(b[:, 1] + b[:, 3] / 2)], 1)
    return np.clip(r, 0, SIZE - 1)


def outside_partials(boxes, partial):
    if len(boxes) == 0 or len(partial) == 0:
        return np.ones(len(boxes), bool)
    p, q = _rects(boxes)[:, None], _rects(partial)[None]
    iw = np.clip(np.minimum(p[..., 1], q[..., 1]) - np.maximum(p[..., 0], q[..., 0]) + 1, 0, None)
    ih = np.clip(np.minimum(p[..., 3], q[..., 3]) - np.maximum(p[..., 2], q[..., 2]) + 1, 0, None)
    area = (p[..., 1] - p[..., 0] + 1) * (p[..., 3] - p[..., 2] + 1)
    return ~(iw * ih / (area + 1e-12) > 0.5).any(1)


def finalize(labels, boxes, scores):
    labels = np.asarray(labels, np.int64)
    boxes, scores = np.asarray(boxes, np.int64).reshape(-1, 4), np.asarray(scores, np.float32)
    keep = [np.flatnonzero(labels == c)[np.argsort(-scores[labels == c], kind="stable")[:WRITE_MAX]]
            for c in np.unique(labels)]
    if not keep:
        return np.array([DUMMY[0]]), np.array([DUMMY[1]]), np.array([DUMMY[2]], np.float32)
    keep = np.concatenate(keep)
    return labels[keep], boxes[keep], scores[keep]


def write_txt(path, dets):
    labels, boxes, scores = finalize(*dets)
    path.write_text("".join(f"{lab} " + " ".join(f"{v / SIZE:.10f}" for v in b) + f" {float(s):.9g}\n"
                            for lab, b, s in zip(labels, boxes, scores)))


def _xyxy(b):
    return box_convert(torch.from_numpy(np.asarray(b, np.float32).reshape(-1, 4)), "cxcywh", "xyxy")


def match(boxes, scores, gt_boxes):
    s, order = torch.sort(torch.from_numpy(np.asarray(scores, np.float32)), descending=True)
    s, order = s[:MAX_DET].numpy(), order[:MAX_DET].numpy()
    thr = IOU_T.numpy()
    tp = np.zeros((len(thr), len(order)), bool)
    if len(order) and len(gt_boxes):
        iou = box_iou(_xyxy(boxes[order]), _xyxy(gt_boxes)).numpy()
        taken, rows = np.zeros((len(thr), len(gt_boxes)), bool), np.arange(len(thr))
        for d in range(len(order)):
            g = iou[d][None, :] * ~taken
            j = g.argmax(1)
            ok = g[rows, j] > thr
            tp[ok, d] = True
            taken[rows[ok], j[ok]] = True
    return s, tp


def image_records(gt, dets, classes):
    labels, boxes, scores = finalize(*dets)
    keep = outside_partials(boxes, gt["partial"])
    return {c: match(boxes[keep & (labels == c)], scores[keep & (labels == c)], gt["boxes"][gt["labels"] == c])
            for c in classes}


def pack(rows):
    assert rows, "no records"
    out = {"imgs": np.array([r[0] for r in rows], np.int64)}
    for c in rows[0][1]:
        out[f"ngt|{c}"] = np.array([r[1][c] for r in rows], np.int64)
    for k, per in rows[0][2].items():
        for c in per:
            out[f"{k}|{c}|s"] = np.concatenate([r[2][k][c][0] for r in rows]).astype(np.float32)
            out[f"{k}|{c}|tp"] = np.concatenate([r[2][k][c][1] for r in rows], 1)
            out[f"{k}|{c}|img"] = np.concatenate([np.full(len(r[2][k][c][0]), r[0], np.int64) for r in rows])
    return out


def class_ap(flat, key, c, imgs=None):
    s, tp, im, ngt = flat[f"{key}|{c}|s"], flat[f"{key}|{c}|tp"], flat[f"{key}|{c}|img"], flat[f"ngt|{c}"]
    if imgs is not None:
        m = np.isin(im, imgs)
        s, tp, ngt = s[m], tp[:, m], ngt[np.isin(flat["imgs"], imgs)]
    n = int(ngt.sum())
    if n == 0:
        return None
    t = torch.from_numpy(tp)[:, torch.argsort(torch.from_numpy(s), descending=True)]
    tp_sum, fp_sum = torch.cumsum(t, 1, dtype=torch.float), torch.cumsum(~t, 1, dtype=torch.float)
    ap = np.zeros(len(IOU_T))
    for k in range(len(IOU_T)):
        if tp_sum.shape[1] == 0:
            continue
        rc = tp_sum[k] / n
        pr = tp_sum[k] / (fp_sum[k] + tp_sum[k] + torch.finfo(torch.float64).eps)
        pr = torch.flip(torch.cummax(torch.flip(pr, [0]), 0).values, [0])
        inds = torch.searchsorted(rc, REC_T, right=False)
        num = int(inds.argmax()) if int(inds.max()) >= len(rc) else len(REC_T)
        prec = torch.zeros(len(REC_T))
        prec[:num] = pr[inds[:num]]
        ap[k] = float(prec.mean())
    return ap


def summary(flat, key, classes, imgs=None):
    aps = {c: class_ap(flat, key, c, imgs) for c in classes}
    v = [a for a in aps.values() if a is not None]
    return {"ap": 100 * float(np.mean([a.mean() for a in v])), "ap_50": 100 * float(np.mean([a[0] for a in v])),
            "ap_75": 100 * float(np.mean([a[5] for a in v])),
            "per_class": {str(c): None if a is None else 100 * float(a.mean()) for c, a in aps.items()}}


REFERENCE = """import json, sys
from pathlib import Path
import torchmetrics
from phenobench.evaluation.evaluate_leaf_bounding_boxes import evaluate_leaf_detection
from phenobench.evaluation.evaluate_plant_bounding_boxes import evaluate_plant_detection
root, pred = Path(sys.argv[1]), Path(sys.argv[2])
out = {"torchmetrics": torchmetrics.__version__}
for task, fn in (("plant", evaluate_plant_detection), ("leaf", evaluate_leaf_detection)):
    if (pred / f"{task}_bboxes").exists():
        out[task] = fn({"phenobench_dir": root, "prediction_dir": pred, "split": "val", "export": pred / "export" / task})
print("RESULT " + json.dumps(out))
"""


def install_reference(side, log):
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--target", side,
                        "torchmetrics==0.10.3", "setuptools<81"], capture_output=True, text=True)
    log(f"torchmetrics 0.10.3 into {side}: {'ok' if r.returncode == 0 else r.stderr[-500:]}")


def reference(root, pred_dir, side, log, timeout):
    script = pred_dir / "reference.py"
    script.write_text(REFERENCE)
    for path in (side, None):
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        if path:
            env["PYTHONPATH"] = path
        try:
            r = subprocess.run([sys.executable, str(script), str(root), str(pred_dir)], env=env, capture_output=True,
                               text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            log(f"dev-kit reference timed out after {timeout} s")
            return None
        lines = [x for x in r.stdout.splitlines() if x.startswith("RESULT ")]
        if r.returncode == 0 and lines:
            return json.loads(lines[-1][7:])
        log(f"dev-kit reference failed (PYTHONPATH {path}): {r.stderr[-800:]}")
    return None


def compare(ours, ref, tag, log):
    gaps = {}
    for task, s in ours.items():
        r = ref[task]
        pairs = [(s["ap"], r["mAP"]), (s["ap_50"], r["mAP_50"]), (s["ap_75"], r["mAP_75"])]
        if "mAP_cls" in r:
            pairs += list(zip(s["per_class"].values(), r["mAP_cls"]))
        gaps[task] = max(abs(a - b) for a, b in pairs)
    tol = 0.006 if ref["torchmetrics"] == "0.10.3" else 0.05
    log(f"{tag}: dev kit with torchmetrics {ref['torchmetrics']}: {ref}; ours "
        f"{ {t: {k: v for k, v in s.items()} for t, s in ours.items()} }; max |ours - devkit| {gaps} (tolerance {tol})")
    assert max(gaps.values()) <= tol, f"AP port disagrees with the dev kit: {gaps}"


def synthetic(inst, sem, vis, rng, n_cls):
    b, lab, _ = instances(inst, sem, vis)
    lab = np.where(lab > 2, lab - 2, lab)
    keep = rng.random(len(b)) > 0.15
    b, lab = b[keep], lab[keep]
    b = b + rng.integers(-3, 4, b.shape) * (rng.random((len(b), 1)) < 0.7)
    if n_cls == 2:
        lab = np.where(rng.random(len(lab)) < 0.15, 3 - lab, lab)
    extra = np.concatenate([rng.integers(40, SIZE - 40, (4, 2)), rng.integers(4, 64, (4, 2))], 1)
    b, lab = np.concatenate([b, extra, b[:3]]), np.concatenate([lab, rng.integers(1, n_cls + 1, 4), lab[:3]])
    b[:, :2] = np.clip(b[:, :2], 0, SIZE - 1)
    b[:, 2:] = np.clip(b[:, 2:], 0, 2 * np.minimum(b[:, :2], SIZE - 1 - b[:, :2]))
    return lab, b, rng.random(len(b)).astype(np.float32)


def check_against_devkit(cfg, root, cache, names, d, side, log):
    from pathlib import Path
    rng = np.random.default_rng(0)
    rows = {"plant": [], "leaf": []}
    tasks = {"plant": ("plant", "sem", "pvis", (1, 2)), "leaf": ("leaf", None, "lvis", (1,))}
    for t in tasks:
        (d / f"{t}_bboxes").mkdir(parents=True, exist_ok=True)
    for k in range(cfg.check_images):
        for t, (ik, sk, vk, cls) in tasks.items():
            inst, vis = np.asarray(cache[f"{ik}_val"][k]), np.asarray(cache[f"{vk}_val"][k])
            sem = None if sk is None else np.asarray(cache[f"{sk}_val"][k])
            gt = ground_truth(inst, sem, vis)
            dets = ([], [], []) if k == 0 else synthetic(inst, sem, vis, rng, len(cls))
            write_txt(d / f"{t}_bboxes" / f"{Path(names[k]).stem}.txt", dets)
            rows[t].append((k, {c: int((gt["labels"] == c).sum()) for c in cls}, {"x": image_records(gt, dets, cls)}))
    ours = {t: summary(pack(rows[t]), "x", tasks[t][3]) for t in tasks}
    ref = reference(root, d, side, log, 900)
    if ref is None:
        log(f"WARNING: no dev-kit reference ran; AP port unchecked on Kaggle (ours on synthetic: {ours})")
        return
    compare(ours, ref, f"metric check vs dev kit ({cfg.check_images} val images, synthetic)", log)
