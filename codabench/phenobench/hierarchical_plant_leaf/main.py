import dataclasses
import json
import time

import numpy as np

from common import runtime as R
from common.data import COUNTS
from common.pq import check_against_devkit, summarize
from common.stitch import find_groups

FOLDERS = ["plant_instances", "leaf_instances"]
WORKER = "hierarchical_plant_leaf.worker"
VAL_KEYS = ["plant_val", "plant_val_maps", "leaf_val", "leaf_stitch", "leaf_val_maps"]
TEST_KEYS = ["plant_test", "plant_test_maps", "leaf_test", "leaf_test_maps"]
log = R.log


def pipeline(cfg):
    out, work, _, names, cache = R.prepare(cfg, log)
    for key, vis in (("plant_val", "pvis_val"), ("leaf_val", "lvis_val")):
        pairs = [(np.asarray(cache[key][i]).astype(np.int32), np.asarray(cache[vis][i])) for i in range(cfg.metric_check_images)]
        check_against_devkit(pairs, lambda msg: log(f"{key}: {msg}"))
    find_groups(cache["img_train"], list(range(COUNTS["train"])), log, "train (check only)")
    pos = R.tile_groups(cache, work, log)
    sidx = {s: R.smoke_idx(pos[s], COUNTS[s], cfg.smoke_images) for s in ("val", "test")}
    R.write_indices(work, sidx)
    cfg = R.with_best_models(cfg, log)
    log(f"smoke run on tiles {sidx}: 2 optimiser steps per zoom specialist, every inference path")
    smoke_res = R.smoke_run(cfg, WORKER, names, sidx, FOLDERS)
    h = R.train_budget(cfg, smoke_res, VAL_KEYS, TEST_KEYS, "training",
                       [("plant_test", "plant_val"), ("leaf_test", "leaf_val")])
    cfg = dataclasses.replace(cfg, train_hours=h)
    log(f"smoke passed; full run: zoom specialists M2F | EoMT {h:.2f} h each on GPU0 | GPU1")
    res = R.full_run(cfg, WORKER, names, FOLDERS)
    dec = res[0]["decision"]
    full = work / "full"
    l_cand = "stitch1" if dec["leaf"]["stitch"] else "stitch0"
    src = "" if dec["plant"]["candidate"] == "v3" else dec["plant"]["candidate"].split("_")[1]
    kept = [f"plant_zoom_{k}" for k, used in (("m2f", src in ("m", "v")), ("eomt", src in ("e", "v"))) if used]
    summary = {
        "val_pq_plant": dec["plant"]["val_pq"], "val_pq_leaf": dec["leaf"]["val_pq"],
        "val_pq": (dec["plant"]["val_pq"] + dec["leaf"]["val_pq"]) / 2, "decision": dec, "tables": res[0]["tables"],
        "val_pq_by_date": {
            "plant": R.by_date(sorted(full.glob("plant_val/c1_*.npz")), names["val"], dec["plant"]["candidate"],
                             dec["plant"]["threshold"], dec["plant"]["offset"], cfg),
            "leaf": R.by_date(sorted(full.glob("leaf_val/c2_*.npz")), names["val"], l_cand, dec["leaf"]["threshold"],
                            dec["leaf"]["offset"], cfg)},
        "diagnostics": {t: summarize(sum((r[f"diag_{t}"]["rows"] for r in res), []), sum((r[f"diag_{t}"]["fp"] for r in res), []))
                        for t in ("plant", "leaf")},
        "train": {"zoom_m2f": res[0]["train_zoom_m2f"], "zoom_eomt": res[1]["train_zoom_eomt"]},
        "timing": {r["rank"]: r["timing"] for r in res},
        "test_instances_median": {t: float(np.median(sum((r[f"{t}_test_instances"] for r in res), []))) for t in ("plant", "leaf")},
        "kept_models": kept, "train_hours": h, "session_hours": (time.time() - R.T0) / 3600}
    (work / "result.json").write_text(json.dumps(summary, indent=1))
    log(json.dumps(summary, indent=1))
    log(f"SUBMIT submission.zip  val PQ {summary['val_pq']:.2f} (plant {summary['val_pq_plant']:.2f}, "
        f"leaf {summary['val_pq_leaf']:.2f}); models kept in results.zip: {kept}")


def run(cfg):
    R.run(cfg, pipeline)
