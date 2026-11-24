import json
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

from common import box_metric as M
from common import runtime as R
from common.data import COUNTS
from leaf import boxes as LD
from plant import boxes as PD

TASKS = {"plant": ("plant_bboxes", PD.CLASSES), "leaf": ("leaf_bboxes", LD.CLASSES)}
WORKER = "detection.worker"
SIDE = "/tmp/tm010"
MAIN_LOG = Path("/tmp/work/logs/main.log")


def log(msg):
    R.log(msg)
    MAIN_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(MAIN_LOG, "a") as f:
        f.write(f"[main {time.strftime('%H:%M:%S')} +{(time.time() - R.T0) / 3600:.2f} h] {msg}\n")


def expected_hours(cfg, smoke_res):
    tm = R.timings(smoke_res)
    nv, nt = (COUNTS["val"] + 1) // 2, (COUNTS["test"] + 1) // 2
    inf = cfg.margin * (nv * sum(v for k, v in tm.items() if k.endswith("_val"))
                        + nt * sum(v for k, v in tm.items() if k.endswith("_test")) + 2 * (nv + nt) * tm["files"])
    decide = cfg.margin * tm["decide_s"] * COUNTS["val"] / cfg.smoke_images
    total = time.time() - R.T0 + 3600 * cfg.yolo_hours + 900 + inf + decide + 600 + 60 * cfg.final_check_minutes
    log(f"smoke s/img: { {k: round(v, 2) for k, v in tm.items()} }; inference {inf / 3600:.2f} h; "
        f"decisions {decide / 3600:.2f} h; YOLO training {cfg.yolo_hours} h; expected session {total / 3600:.2f} h")
    assert total < cfg.session_hours * 3600, f"expected session {total / 3600:.2f} h exceeds {cfg.session_hours} h"


def package(pred_dir, folder, names, labels, zip_path):
    files = [Path(n).stem + ".txt" for n in names]
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(folder + "/", "")
        for f in files:
            z.write(pred_dir / folder / f, f"{folder}/{f}")
    n_box = 0
    with zipfile.ZipFile(zip_path) as z:
        expected = {folder + "/"} | {f"{folder}/{f}" for f in files}
        assert set(z.namelist()) == expected, f"zip entries differ: {sorted(set(z.namelist()) ^ expected)[:5]}"
        for f in files:
            rows = [line.split() for line in z.read(f"{folder}/{f}").decode().splitlines()]
            assert rows and all(len(r) == 6 for r in rows), f"{f}: empty or malformed"
            for r in rows:
                v = [float(x) for x in r[1:]]
                assert int(r[0]) in labels and all(0 <= x <= 1 for x in v), f"{f}: bad line {r}"
            n_box += len(rows)
    log(f"validated {zip_path}: {len(files)} files in {folder}/, {n_box / len(files):.1f} boxes per image, "
        f"{zip_path.stat().st_size / 2**20:.1f} MB")


def final_check(cfg, root, names, work, dec):
    n, d = cfg.final_check_images, work / "check" / "final"
    for folder, _ in TASKS.values():
        (d / folder).mkdir(parents=True, exist_ok=True)
        for k in range(n):
            f = Path(names[k]).stem + ".txt"
            shutil.copy(work / "full" / "val_pred" / folder / f, d / folder / f)
    ours = {t: dec[t]["first_n"] for t in TASKS}
    ref = M.reference(root, d, SIDE, log, 60 * cfg.final_check_minutes)
    if ref is None:
        log(f"WARNING: final dev-kit check did not run; ours on the first {n} val images: {ours}")
        return
    M.compare(ours, ref, f"final check vs dev kit (chosen arms, first {n} val images)", log)


def pipeline(cfg):
    out, work, root, names, cache = R.prepare(cfg, log)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics"], check=True)
    M.install_reference(SIDE, log)
    M.check_against_devkit(cfg, root, cache, names["val"], work / "check" / "synthetic", SIDE, log)
    sidx = {s: list(range(cfg.smoke_images)) for s in ("val", "test")}
    R.write_indices(work, sidx)
    cfg = R.with_best_models(cfg, log)
    log("smoke run: every path on the smoke images")
    smoke_res = R.stage(cfg, "smoke", WORKER)
    for t, (folder, cls) in TASKS.items():
        package(work / "smoke" / "pred", folder, [names["test"][i] for i in sidx["test"]], cls,
                work / "smoke" / f"submission_{t}_smoke.zip")
    expected_hours(cfg, smoke_res)
    log("smoke passed; full run")
    res = R.stage(cfg, "full", WORKER)
    for t, (folder, cls) in TASKS.items():
        package(work / "full" / "pred", folder, names["test"], cls, out / f"submission_{t}.zip")
    dec = res[0]["decision"]
    for t in TASKS:
        (work / "full" / "models" / f"yolo_{t}").mkdir(parents=True, exist_ok=True)
        shutil.copy(work / "full" / "yolo" / t / "run" / "weights" / "last.pt", work / "full" / "models" / f"yolo_{t}" / "last.pt")
    summary = {"val": {t: {k: v for k, v in d.items() if k != "table"} for t, d in dec.items()},
               "tables": {t: d["table"] for t, d in dec.items()}, "timing": {r["rank"]: r["timing"] for r in res},
               "kept_models": [f"yolo_{t}" for t in TASKS], "session_hours": (time.time() - R.T0) / 3600}
    (work / "result.json").write_text(json.dumps(summary, indent=1))
    log(json.dumps({k: v for k, v in summary.items() if k != "tables"}, indent=1))
    p, lf = dec["plant"], dec["leaf"]
    log(f"SUBMIT submission_plant.zip (plant track)  val mAP {p['ap']:.2f} (AP crop {p['per_class']['1']:.2f}, "
        f"AP weed {p['per_class']['2']:.2f}, mAP_50 {p['ap_50']:.2f}, mAP_75 {p['ap_75']:.2f}); {p['key']}")
    log(f"SUBMIT submission_leaf.zip (leaf track)  val AP {lf['ap']:.2f} (AP_50 {lf['ap_50']:.2f}, "
        f"AP_75 {lf['ap_75']:.2f}); {lf['key']}")
    final_check(cfg, root, names["val"], work, dec)


def run(cfg):
    R.run(cfg, pipeline, log)
