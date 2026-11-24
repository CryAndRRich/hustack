import numpy as np

from common.pq import pq_curve, synthetic_prediction

MERGE = np.array([0, 1, 2, 1, 2], np.int64)


def crop_gt(plant, sem, vis):
    return np.where((sem == 1) | (vis <= 127), plant, 0)


def confusion(gt_sem, pred_sem):
    return np.bincount(MERGE[gt_sem.ravel()] * 3 + MERGE[pred_sem.ravel()], minlength=9).reshape(3, 3)


def ious(conf):
    tp = np.diag(conf).astype(np.float64)
    return 100 * tp / np.maximum(conf.sum(0) + conf.sum(1) - tp, 1)


def synthetic_classes(plant, sem, vis, rng):
    import torch
    from phenobench.evaluation.auxiliary.filter import filter_partial_masks
    from phenobench.evaluation.auxiliary.panoptic_eval import PanopticQuality
    pred = synthetic_prediction(plant, rng)
    cls = rng.integers(1, 3, int(pred.max()) + 1)
    psem = np.where(pred > 0, cls[pred], 0)
    gt_t, gs_t = torch.from_numpy(plant.astype(np.int32)), torch.from_numpy(sem.astype(np.int32))
    pred_t, ps_t = torch.from_numpy(pred.astype(np.int32)), torch.from_numpy(psem.astype(np.int32))
    filter_partial_masks(pred_t, ps_t, gt_t, gs_t, torch.from_numpy(vis.astype(np.float32) / 255.0))
    pq = PanopticQuality()
    pq.compute_pq(ps_t, gs_t, pred_t, gt_t)
    return pred, psem, pq.panoptic_qualities


def check_against_devkit(triples, log):
    import torch
    from torchmetrics.classification import MulticlassJaccardIndex
    rng = np.random.default_rng(0)
    jac, conf, worst = MulticlassJaccardIndex(num_classes=3, average=None), np.zeros((3, 3), np.int64), 0.0
    for plant, sem, vis in triples:
        pred, psem, ref_pq = synthetic_classes(plant, sem, vis, rng)
        ours = pq_curve(np.where(psem == 1, pred, 0), np.ones(int(pred.max())), crop_gt(plant, sem, vis), vis, None, [0.0])[0, 0]
        ref = ref_pq[1]["pq"] if 1 in ref_pq else float("nan")
        if not (np.isnan(ours) and np.isnan(ref)):
            worst = max(worst, abs(ours - ref))
        jac.update(torch.from_numpy(MERGE[psem]), torch.from_numpy(MERGE[sem.astype(np.int64)]))
        conf += confusion(sem, psem)
    iou_gap = float(np.abs(ious(conf) - jac.compute().numpy() * 100).max())
    log(f"metric check vs dev kit: {len(triples)} images, PQ(crop) max |ours - devkit| = {worst:.2e}, "
        f"IoU max gap {iou_gap:.2e}")
    assert worst < 1e-4 and iou_gap < 1e-3, f"metric port disagrees with the dev kit: PQ {worst}, IoU {iou_gap}"
