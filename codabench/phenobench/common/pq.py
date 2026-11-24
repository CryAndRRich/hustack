import numpy as np


def _match(pred, gt, vis, dont_care):
    p, g = pred.ravel().astype(np.int64), gt.ravel().astype(np.int64)
    gid, ginv = np.unique(g, return_inverse=True)
    vis_of = np.zeros(gid.size, np.int64)
    vis_of[ginv] = vis.ravel()
    valid = (gid > 0) & (vis_of > 127)
    partial = (gid > 0) & (vis_of <= 127)
    if not valid.any():
        return None
    n = int(p.max()) + 1
    cont = np.bincount(p * gid.size + ginv, minlength=n * gid.size).reshape(n, gid.size)
    area = cont.sum(1)
    present = (np.arange(n) > 0) & (area > 0)
    safe = np.maximum(area, 1)
    removed = np.zeros(n, bool)
    if partial.any():
        removed |= (cont[:, partial] / safe[:, None] > 0.5).any(1)
    if dont_care is not None:
        removed |= np.bincount(p[dont_care.ravel()], minlength=n) / safe > 0.5
    keep = present & ~removed
    inter = cont[:, valid].astype(np.float64)
    union = area[:, None] + cont.sum(0)[valid][None, :] - inter
    iou = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
    return {"gid": gid[valid], "keep": keep, "iou": iou, "area": area, "gt_area": cont.sum(0)[valid]}


def pq_curve(pred, scores, gt, vis, dont_care, thresholds, offsets=(0.0,), split=0):
    t = _match(pred, gt, vis, dont_care)
    if t is None:
        return np.full((len(offsets), len(thresholds)), np.nan)
    n = len(t["keep"])
    best = np.where(t["iou"] > 0.5, t["iou"], 0.0).max(1)
    sc = np.zeros(n)
    sc[1:] = scores[:n - 1]
    small = t["area"] < split
    n_gt = len(t["gid"])
    out = np.zeros((len(offsets), len(thresholds)))
    for a, off in enumerate(offsets):
        for b, th in enumerate(thresholds):
            k = t["keep"] & (sc >= np.where(small, th + off, th))
            tp = int((best[k] > 0).sum())
            fp = int(k.sum()) - tp
            out[a, b] = best[k].sum() / (tp + 0.5 * fp + 0.5 * (n_gt - tp))
    return out


BINS = [0, 500, 2500, 10000, 10 ** 9]


def instance_stats(pred, gt, vis, dont_care, sem=None):
    t = _match(pred, gt, vis, dont_care)
    if t is None:
        return [], []
    iou = np.where(t["iou"] > 0.5, t["iou"], 0.0) * t["keep"][:, None]
    matched = iou.max(0)
    crop = np.full(len(t["gid"]), -1)
    if sem is not None:
        g, s = gt.ravel().astype(np.int64), sem.ravel().astype(np.int64)
        for k, i in enumerate(t["gid"]):
            v = np.bincount(s[g == i], minlength=5)
            crop[k] = int(v[1] + v[3] >= v[2] + v[4])
    pred_area = np.where(matched > 0, t["area"][iou.argmax(0)], 0)
    fp = t["keep"] & (iou.max(1) == 0)
    return list(zip(t["gt_area"].tolist(), matched.tolist(), crop.tolist(), pred_area.tolist())), t["area"][fp].tolist()


def summarize(gt_rows, fp_areas):
    a = np.array(gt_rows, dtype=np.float64).reshape(-1, 5)[:, 1:]
    fp = np.array(fp_areas, dtype=np.float64)
    out = {}
    for name, sel in (("all", np.ones(len(a), bool)), ("crop", a[:, 2] == 1), ("weed", a[:, 2] == 0)):
        if not sel.any():
            continue
        for lo, hi in zip(BINS[:-1], BINS[1:]):
            b = sel & (a[:, 0] >= lo) & (a[:, 0] < hi)
            tp = b & (a[:, 1] > 0)
            out[f"{name} {lo}-{hi}"] = {"n_gt": int(b.sum()), "recall": float(tp.sum() / max(b.sum(), 1)),
                                         "sq": float(a[tp, 1].mean()) if tp.any() else None,
                                         "area_ratio_median": float(np.median(a[tp, 3] / a[tp, 0])) if tp.any() else None}
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        out[f"fp {lo}-{hi}"] = int(((fp >= lo) & (fp < hi)).sum())
    return out


def leaf_dont_care(plant_ids, leaf_ids):
    return (plant_ids > 0) & (leaf_ids == 0)


def synthetic_prediction(gt, rng):
    pred = np.zeros_like(gt)
    ids = [i for i in np.unique(gt) if i > 0]
    nxt = 1
    for i in ids:
        r = rng.random()
        if r < 0.15:
            continue
        m = gt == i
        if r < 0.35:
            m = np.roll(m, tuple(rng.integers(-8, 9, 2)), axis=(0, 1))
        pred[m & (pred == 0)] = nxt if r >= 0.45 or nxt == 1 else nxt - 1
        nxt += 1
    for _ in range(3):
        y, x = rng.integers(0, gt.shape[0] - 64, 2)
        s = int(rng.integers(8, 64))
        pred[y:y + s, x:x + s][pred[y:y + s, x:x + s] == 0] = nxt
        nxt += 1
    return pred


def devkit_pq(pred, gt, vis):
    import torch
    from phenobench.evaluation.auxiliary.filter import filter_partial_masks
    from phenobench.evaluation.auxiliary.panoptic_eval import PanopticQuality
    gt_t = torch.from_numpy(gt.astype(np.int32))
    pred_t = torch.from_numpy(pred.astype(np.int32))
    vis_t = torch.from_numpy(vis.astype(np.float32) / 255.0)
    gt_sem = (gt_t > 0).type(torch.uint8)
    pred_sem = (pred_t > 0).type(torch.uint8)
    filter_partial_masks(pred_t, pred_sem, gt_t, gt_sem, vis_t)
    pq = PanopticQuality()
    pq.compute_pq(pred_sem, gt_sem, pred_t, gt_t)
    return pq.panoptic_qualities[1]["pq"] if 1 in pq.panoptic_qualities else float("nan")


def check_against_devkit(pairs, log):
    rng = np.random.default_rng(0)
    worst = 0.0
    for gt, vis in pairs:
        pred = synthetic_prediction(gt, rng)
        ours = pq_curve(pred, np.ones(int(pred.max())), gt, vis, None, [0.0])[0, 0]
        ref = devkit_pq(pred, gt, vis)
        if np.isnan(ours) and np.isnan(ref):
            continue
        worst = max(worst, abs(ours - ref))
    log(f"metric check vs dev kit: {len(pairs)} images, max |ours - devkit| = {worst:.2e}")
    assert worst < 1e-4, f"metric port disagrees with the dev kit by {worst}"
