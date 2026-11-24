import dataclasses
import json
import time

import numpy as np

from common import runtime as R
from common.cropcls import save_labels
from common.data import COUNTS
from common.pq import check_against_devkit, summarize
from common.sem_metric import check_against_devkit as check_hierarchical

FOLDERS = ["semantics", "plant_instances", "leaf_instances"]
WORKER = "hierarchical_panoptic.worker"
VAL_KEYS = ["plant_val", "crop_val", "sem_val", "leaf_val", "leaf_stitch", "leaf_val_maps"]
TEST_KEYS = ["plant_test", "crop_test", "sem_test", "leaf_test", "leaf_test_maps"]
log = R.log


def pipeline(cfg):
    out, work, _, names, cache = R.prepare(cfg, log)
    n = cfg.metric_check_images
    check_against_devkit([(np.asarray(cache["leaf_val"][i]).astype(np.int32), np.asarray(cache["lvis_val"][i])) for i in range(n)],
                         lambda msg: log(f"PQ(leaf): {msg}"))
    check_hierarchical([(np.asarray(cache["plant_val"][i]).astype(np.int32), np.asarray(cache["sem_val"][i]),
                         np.asarray(cache["pvis_val"][i])) for i in range(n)], lambda msg: log(f"PQ(crop) + IoU: {msg}"))
    save_labels(work / "cache", cache, log)
    pos = R.tile_groups(cache, work, log)
    sidx = {s: R.smoke_idx(pos[s], COUNTS[s], cfg.smoke_images) for s in ("val", "test")}
    R.write_indices(work, sidx)
    cfg = R.with_best_models(cfg, log)
    log(f"smoke tiles {sidx}")
    log("smoke run: 2 optimiser steps per semantic model, every inference path on the smoke tiles")
    smoke_res = R.smoke_run(cfg, WORKER, names, sidx, FOLDERS)
    h = R.train_budget(cfg, smoke_res, VAL_KEYS, TEST_KEYS, "semantic training",
                       [("sem_test", "sem_val")])
    cfg = dataclasses.replace(cfg, train_hours=h)
    log(f"smoke passed; full run: semantic models {h:.2f} h each on GPU0 | GPU1")
    res = R.full_run(cfg, WORKER, names, FOLDERS)
    dec = res[0]["decision"]
    sem = dec["semantics"]
    val = {"pq_crop": dec["plant"]["val_pq"], "pq_leaf": dec["leaf"]["val_pq"], "iou_weed": sem["iou_weed"],
           "iou_soil": sem["iou_soil"]}
    val["pq_plus"] = float(np.mean(list(val.values())))
    src = sem["candidate"][0] or ""
    kept = [f"sem_{r}" for r in range(2) if src.startswith(("ens", f"m{r}"))]
    full = work / "full"
    summary = {
        "val": val, "decision": dec, "tables": res[0]["tables"],
        "val_pq_by_date": {
            "crop": R.by_date(sorted(full.glob("plant_val/c1_*.npz")), names["val"], f"c{dec['plant']['crop_threshold']}",
                              dec["plant"]["threshold"], dec["plant"]["offset"], cfg),
            "leaf": R.by_date(sorted(full.glob("leaf_val/c2_*.npz")), names["val"], "stitch1" if dec["leaf"]["stitch"] else "stitch0",
                              dec["leaf"]["threshold"], dec["leaf"]["offset"], cfg)},
        "diagnostics": {t: summarize(sum((r[f"diag_{t}"]["rows"] for r in res), []), sum((r[f"diag_{t}"]["fp"] for r in res), []))
                        for t in ("plant", "leaf")},
        "train_sem": [r["train_sem"] for r in res], "timing": {r["rank"]: r["timing"] for r in res},
        "test_leaf_instances_median": float(np.median(sum((r["leaf_test_instances"] for r in res), []))),
        "kept_models": kept, "train_hours": h, "session_hours": (time.time() - R.T0) / 3600}
    (work / "result.json").write_text(json.dumps(summary, indent=1))
    log(json.dumps(summary, indent=1))
    log(f"SUBMIT submission.zip  val PQ+ {val['pq_plus']:.2f} (PQ crop {val['pq_crop']:.2f}, PQ leaf {val['pq_leaf']:.2f}, "
        f"IoU weed {val['iou_weed']:.2f}, IoU soil {val['iou_soil']:.2f}); semantic models kept in results.zip: {kept}")


def run(cfg):
    R.run(cfg, pipeline)
