import json
import time

import numpy as np

from common import runtime as R
from common.cropcls import save_labels
from common.data import COUNTS
from common.pq import check_against_devkit, summarize

FOLDERS = ["leaf_instances"]
WORKER = "leaf_instance.worker"
log = R.log


def estimate(smoke_res):
    tm = R.timings(smoke_res)
    nv, nt = (COUNTS["val"] + 1) // 2, (COUNTS["test"] + 1) // 2
    val_keys = ["plant_val", "crop_val", "leaf_val", "leaf_stitch", "leaf_val_maps"]
    test_keys = ["plant_test", "crop_test", "leaf_test", "leaf_test_maps"]
    inf = nv * sum(tm[k] for k in val_keys) + nt * sum(tm[k] for k in test_keys)
    log(f"smoke s/img: { {k: round(v, 2) for k, v in tm.items()} }; inference {inf / 3600:.2f} h")
    return inf / 3600


def pipeline(cfg):
    out, work, _, names, cache = R.prepare(cfg, log)
    pairs = [(np.asarray(cache["leaf_val"][i]).astype(np.int32), np.asarray(cache["lvis_val"][i])) for i in range(cfg.metric_check_images)]
    check_against_devkit(pairs, lambda msg: log(f"leaf_val (leaf_instances metric, no don't-care): {msg}"))
    save_labels(work / "cache", cache, log)
    pos = R.tile_groups(cache, work, log)
    sidx = {s: R.smoke_idx(pos[s], COUNTS[s], cfg.smoke_images) for s in ("val", "test")}
    R.write_indices(work, sidx)
    cfg = R.with_best_models(cfg, log)
    log(f"smoke tiles {sidx}")
    log("smoke run: 2 classifier steps per GPU, every inference path on the smoke tiles")
    smoke_res = R.smoke_run(cfg, WORKER, names, sidx, FOLDERS)
    total = (time.time() - R.T0) / 3600 + cfg.cls_hours + cfg.margin * estimate(smoke_res) + 0.25
    log(f"smoke passed; full run: crop classifier {cfg.cls_hours:.2f} h per GPU; expected session {total:.2f} h")
    assert total < cfg.session_hours, f"expected session {total:.2f} h exceeds {cfg.session_hours} h"
    res = R.full_run(cfg, WORKER, names, FOLDERS)
    dec = res[0]["decision"]
    full = work / "full"
    summary = {
        "val_pq_leaf": dec["val_pq"], "decision": dec, "tables": res[0]["tables"],
        "val_pq_by_date": R.by_date(sorted(full.glob("leaf_val/c2_*.npz")), names["val"],
                                  "stitch1" if dec["stitch"] else "stitch0", dec["threshold"], dec["offset"], cfg),
        "diagnostics": summarize(sum((r["diag_leaf"]["rows"] for r in res), []), sum((r["diag_leaf"]["fp"] for r in res), [])),
        "crop_classifier": {"train": [r["train_cls"] for r in res], "ensemble_val": res[0]["cls_val_ensemble"]},
        "timing": {r["rank"]: r["timing"] for r in res},
        "test_instances_median": float(np.median(sum((r["leaf_test_instances"] for r in res), []))),
        "kept_models": ["crop_cls"], "session_hours": (time.time() - R.T0) / 3600}
    (work / "result.json").write_text(json.dumps(summary, indent=1))
    log(json.dumps(summary, indent=1))
    log(f"SUBMIT submission.zip  val PQ(leaf) {dec['val_pq']:.2f}; crop classifiers kept in results.zip")


def run(cfg):
    R.run(cfg, pipeline)
