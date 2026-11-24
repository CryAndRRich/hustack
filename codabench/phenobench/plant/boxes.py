import numpy as np

from common import leaf
from common.box_metric import instances, mask_boxes

TASK = "plant"
CLASSES = (1, 2)
NAMES = {0: "crop", 1: "weed"}


def yolo_labels(cache, i):
    inst = np.asarray(cache["plant_train"][i])
    b, lab, _ = instances(inst, np.asarray(cache["sem_train"][i]), inst)
    return np.where(lab > 2, lab - 2, lab) - 1, b


def ground_truth(cache, i, gt_fn):
    return gt_fn(*(np.asarray(cache[f"{c}_val"][i]) for c in ("plant", "sem", "pvis")))


def mask_source(dirs, split, i):
    pm, ps = leaf.plant_map(dirs[f"maps_{split}"], i)
    pc = np.load(dirs[f"crop_{split}"] / f"{i}.npy")
    boxes, ok = mask_boxes(pm)
    b, s, p = boxes[ok], ps[ok], pc[ok]
    sq = np.sqrt(s)
    return (np.repeat([1, 2], len(b)), np.concatenate([b, b]), np.concatenate([sq * p, sq * (1 - p)]),
            {"plant": np.concatenate([s, s]), "pcrop": np.concatenate([p, p])})
