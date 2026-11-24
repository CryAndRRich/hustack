import dataclasses
import json
import time

import numpy as np

from common import runtime as R
from common.cropcls import save_labels
from common.sem_metric import check_against_devkit

FOLDERS = ["semantics"]
WORKER = "semantic.worker"
VAL_KEYS = ["plant_val", "crop_val", "sem_val"]
TEST_KEYS = ["plant_test", "crop_test", "sem_test"]
log = R.log


def pipeline(cfg):
    out, work, _, names, cache = R.prepare(cfg, log)
    n = cfg.metric_check_images
    check_against_devkit([(np.asarray(cache["plant_val"][i]).astype(np.int32), np.asarray(cache["sem_val"][i]),
                           np.asarray(cache["pvis_val"][i])) for i in range(n)], lambda msg: log(f"IoU (+ PQ crop): {msg}"))
    save_labels(work / "cache", cache, log)
    sidx = {s: list(range(cfg.smoke_images)) for s in ("val", "test")}
    R.write_indices(work, sidx)
    cfg = R.with_best_models(cfg, log)
    log("smoke run: 2 optimiser steps per semantic model, every inference path on the smoke images")
    smoke_res = R.smoke_run(cfg, WORKER, names, sidx, FOLDERS)
    h = R.train_budget(cfg, smoke_res, VAL_KEYS, TEST_KEYS, "semantic training",
                       [("sem_test", "sem_val")])
    cfg = dataclasses.replace(cfg, train_hours=h)
    log(f"smoke passed; full run: semantic models {h:.2f} h each on GPU0 | GPU1")
    res = R.full_run(cfg, WORKER, names, FOLDERS)
    dec = res[0]["decision"]
    kept = [f"sem_{m}" for m in dec["candidate"][0] if m != "a"]
    summary = {"val": {k: dec[k] for k in ("miou", "iou_soil", "iou_crop", "iou_weed")}, "decision": dec,
               "table": res[0]["table"], "train_sem": {s["name"]: r["train_sem"] for s, r in zip(cfg.sem_train, res)},
               "timing": {r["rank"]: r["timing"] for r in res}, "kept_models": kept, "train_hours": h,
               "session_hours": (time.time() - R.T0) / 3600}
    (work / "result.json").write_text(json.dumps(summary, indent=1))
    log(json.dumps({k: v for k, v in summary.items() if k != "table"}, indent=1))
    log(f"SUBMIT submission.zip  val mIoU {dec['miou']:.2f} (soil {dec['iou_soil']:.2f}, crop {dec['iou_crop']:.2f}, "
        f"weed {dec['iou_weed']:.2f}); {dec['name']}; models kept in results.zip: {kept}")


def run(cfg):
    R.run(cfg, pipeline)
