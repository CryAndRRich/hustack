import json
import time

import numpy as np

from common import runtime as R
from common.data import COUNTS
from common.sem_metric import check_against_devkit as check_iou, ious
from panoptic.metric import check_against_devkit

FOLDERS = ["semantics", "plant_instances"]
WORKER = "panoptic.worker"
WEED_BINS = [0, 100, 500, 10 ** 9]
log = R.log


def expected_hours(cfg, smoke_res):
    tm = R.timings(smoke_res)
    nv, nt = (COUNTS["val"] + 1) // 2, (COUNTS["test"] + 1) // 2
    inf = cfg.margin * sum(n * sum(v for k, v in tm.items() if k.endswith(s)) for n, s in ((nv, "_val"), (nt, "_test")))
    total = time.time() - R.T0 + inf + 600
    log(f"smoke s/img: { {k: round(v, 2) for k, v in tm.items()} }; inference {inf / 3600:.2f} h; "
        f"expected session {total / 3600:.2f} h")
    assert total < cfg.session_hours * 3600, f"expected session {total / 3600:.2f} h exceeds {cfg.session_hours} h"


def weed_summary(rows, fps):
    a = np.array(rows, dtype=np.float64).reshape(-1, 5)
    fp = np.array(fps, dtype=np.float64)
    out = {}
    for lo, hi in zip(WEED_BINS[:-1], WEED_BINS[1:]):
        b = (a[:, 1] >= lo) & (a[:, 1] < hi)
        tp = b & (a[:, 2] > 0)
        out[f"{lo}-{hi}"] = {"n_gt": int(b.sum()), "tp": int(tp.sum()), "recall": float(tp.sum() / max(b.sum(), 1)),
                             "sq": float(a[tp, 2].mean()) if tp.any() else None,
                             "fp": int(((fp >= lo) & (fp < hi)).sum())}
    return out


def steps(table, dec):
    ok = {k: v["val_pq"] for k, v in table.items() if float(k[1:].split("_r")[0]) <= dec["crop"]["crop_threshold"]}
    return {"plants": max(v for k, v in ok.items() if "_rNone_t0" in k),
            "plants+recovered": max(v for k, v in ok.items() if k.endswith("_t0")),
            "plants+recovered+trim": dec["weed"]["val_pq"]}


def pipeline(cfg):
    out, work, _, names, cache = R.prepare(cfg, log)
    triples = [(np.asarray(cache["plant_val"][i]).astype(np.int32), np.asarray(cache["sem_val"][i]),
                np.asarray(cache["pvis_val"][i])) for i in range(cfg.metric_check_images)]
    check_against_devkit(triples, lambda msg: log(f"PQ(crop), PQ(weed): {msg}"))
    check_iou(triples, lambda msg: log(f"IoU: {msg}"))
    sidx = {s: list(range(cfg.smoke_images)) for s in ("val", "test")}
    R.write_indices(work, sidx)
    cfg = R.with_best_models(cfg, log)
    log("smoke run: every path on the smoke images")
    smoke_res = R.smoke_run(cfg, WORKER, names, sidx, FOLDERS)
    expected_hours(cfg, smoke_res)
    log("smoke passed; full run")
    res = R.full_run(cfg, WORKER, names, FOLDERS)
    dec, tables = res[0]["decision"], res[0]["tables"]
    soil, crop_iou, weed_iou = ious(sum(np.array(r["val"]["conf"]) for r in res))
    val = {"pq_crop": dec["crop"]["val_pq"], "pq_weed": dec["weed"]["val_pq"], "iou_soil": soil}
    val["pq_plus"] = float(np.mean(list(val.values())))
    pv = sorted((work / "full" / "plant_val").glob("curves_*.npz"))
    summary = {
        "val": val, "val_iou_crop_weed": [crop_iou, weed_iou], "decision": dec, "weed_steps": steps(tables["weed"], dec),
        "val_pq_by_date": {
            "crop": R.by_date([p for p in pv if p.name.startswith("curves_crop_")], names["val"],
                              f"c{dec['crop']['crop_threshold']}", dec["crop"]["threshold"], dec["crop"]["offset"], cfg),
            "weed": R.by_date([p for p in pv if p.name.startswith("curves_weed_")], names["val"], dec["weed"]["key"],
                              dec["weed"]["threshold"], dec["weed"]["offset"], cfg)},
        "weed_diagnostics": weed_summary(sum((r["val"]["rows"] for r in res), []), sum((r["val"]["fp"] for r in res), [])),
        "tables": tables, "timing": {r["rank"]: r["timing"] for r in res}, "kept_models": [],
        "session_hours": (time.time() - R.T0) / 3600}
    (work / "result.json").write_text(json.dumps(summary, indent=1))
    log(json.dumps({k: v for k, v in summary.items() if k != "tables"}, indent=1))
    log(f"SUBMIT submission.zip  val PQ+ {val['pq_plus']:.2f} (PQ crop {val['pq_crop']:.2f}, PQ weed {val['pq_weed']:.2f}, "
        f"IoU soil {soil:.2f}); weeds {dec['weed']['key']}")


def run(cfg):
    R.run(cfg, pipeline)
