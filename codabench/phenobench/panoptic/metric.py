import numpy as np

from common.pq import _match
from common.sem_metric import synthetic_classes


def class_gt(plant, sem, vis, k):
    return np.where((sem == k) | (vis <= 127), plant, 0)


def pq_curves(pred, score_sets, gt, vis, thresholds, offsets, split):
    t = _match(pred, gt, vis, None)
    if t is None:
        return {key: np.full((len(offsets), len(thresholds)), np.nan) for key in score_sets}
    n = len(t["keep"])
    best = np.where(t["iou"] > 0.5, t["iou"], 0.0).max(1)
    small = t["area"] < split
    th = np.asarray(thresholds, np.float64)[None, :, None]
    lim = np.where(small, th + np.asarray(offsets, np.float64)[:, None, None], th)
    n_gt = len(t["gid"])
    out = {}
    for key, scores in score_sets.items():
        sc = np.zeros(n)
        sc[1:] = scores[:n - 1]
        k = t["keep"] & (sc >= lim)
        tp = (k & (best > 0)).sum(-1)
        out[key] = (k * best).sum(-1) / (tp + 0.5 * (k.sum(-1) - tp) + 0.5 * (n_gt - tp))
    return out


def check_against_devkit(triples, log):
    rng = np.random.default_rng(1)
    worst = {1: 0.0, 2: 0.0}
    for plant, sem, vis in triples:
        pred, psem, ref_pq = synthetic_classes(plant, sem, vis, rng)
        for k in (1, 2):
            ours = pq_curves(np.where(psem == k, pred, 0), {"x": np.ones(int(pred.max()))}, class_gt(plant, sem, vis, k),
                             vis, [0.0], [0.0], 0)["x"][0, 0]
            ref = ref_pq[k]["pq"] if k in ref_pq else float("nan")
            if np.isnan(ours) != np.isnan(ref):
                worst[k] = float("inf")
            elif not np.isnan(ours):
                worst[k] = max(worst[k], abs(ours - ref))
    log(f"metric check vs dev kit: {len(triples)} images, max |ours - devkit| PQ(crop) {worst[1]:.2e}, "
        f"PQ(weed) {worst[2]:.2e}")
    assert max(worst.values()) < 1e-4, f"metric port disagrees with the dev kit: {worst}"
