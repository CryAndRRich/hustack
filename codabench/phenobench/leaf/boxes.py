import numpy as np

from common import items as I
from common.box_metric import instances, mask_boxes

TASK = "leaf"
CLASSES = (1,)
NAMES = {0: "leaf"}


def yolo_labels(cache, i):
    inst = np.asarray(cache["leaf_train"][i])
    b, _, _ = instances(inst, None, inst)
    return np.zeros(len(b), np.int64), b


def ground_truth(cache, i, gt_fn):
    return gt_fn(np.asarray(cache["leaf_val"][i]), None, np.asarray(cache["lvis_val"][i]))


def features(cfg, its, pc):
    canvas, kept, items = I.paint(its, cfg.keep_frac, cfg.min_score)
    return {"b": mask_boxes(canvas)[0], "l": np.asarray(kept, np.float64),
            "s": np.array([it[4][0] for it in items], np.float64),
            "c": pc[np.array([it[4][1][1] for it in items], np.int64) - 1]}


def mask_source(dirs, split, i):
    f = np.load(dirs[f"leaf_{split}"] / f"{i}.npz")
    return np.ones(len(f["l"]), np.int64), f["b"], f["l"] * f["c"], {"leaf": f["l"], "plant": f["s"], "pcrop": f["c"]}
