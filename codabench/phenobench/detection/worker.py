import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

from common import box_metric as M
from common import box_select as S
from common import leaf
from common import sem_model as SM
from common import yolo as Y
from common.data import COUNTS
from common.models import load_eomt
from common.plant import image_tensor
from common.runtime import Worker, per_image
from common.stages import plant_crop_stage
from detection.config import Config
from leaf import boxes as LD
from plant import boxes as PD

TRACKS = (PD, LD)


def leaf_pass(cfg, model, cache, split, shard, dirs, dev, log):
    times = []
    for k, i in enumerate(shard):
        t = time.time()
        pm, ps = leaf.plant_map(dirs[f"maps_{split}"], i)
        pc = np.load(dirs[f"crop_{split}"] / f"{i}.npy")
        lm = np.where(np.concatenate([[0.0], ps])[pm] >= cfg.leaf_plant_gate, pm, 0)
        its = leaf.crop_items(cfg, model, np.asarray(cache[f"img_{split}"][i]), lm, ps, dev, i)
        np.savez_compressed(dirs[f"leaf_{split}"] / f"{i}.npz", **LD.features(cfg, its, pc))
        times.append(time.time() - t)
        if k % 50 == 0:
            log(f"leaf {split} {k}/{len(shard)} {np.sum(times):.0f} s")
    return times


def sem_pass(cfg, model, cache, split, shard, dirs, dev):
    times = []
    for i in shard:
        t = time.time()
        p = SM.view(model, image_tensor(cache[f"img_{split}"][i], dev), *cfg.sem_view, cfg.tile_batch)
        np.savez_compressed(dirs[f"sem_{split}"] / f"{i}.npz", lab=p.argmax(0).cpu().numpy().astype(np.uint8))
        times.append(time.time() - t)
    return times


def yolo_pass(cfg, models, cache, split, shard, dirs):
    times = []
    for i in shard:
        t = time.time()
        img = np.asarray(cache[f"img_{split}"][i])
        for task, m in models.items():
            r = Y.predict(m, img, cfg.yolo_views, cfg.yolo_conf, cfg.yolo_max_det)
            np.savez_compressed(dirs[f"yolo_{split}"] / f"{task}_{i}.npz",
                                **{f"{v}|{k}": a for v, (c, b, s) in r.items() for k, a in zip("cbs", (c, b, s))})
        times.append(time.time() - t)
    return times


DIRS = [f"{a}_{s}" for a in ("plant", "maps", "crop", "leaf", "sem", "yolo") for s in ("val", "test")] + [
    "pred/plant_bboxes", "pred/leaf_bboxes", "val_pred/plant_bboxes", "val_pred/leaf_bboxes"]


def main(rank, mode, cfg_path):
    w = Worker(Config, rank, mode, cfg_path, DIRS)
    rank, cfg, smoke, log, out, names, idx = w.rank, w.cfg, w.smoke, w.log, w.out, w.names, w.idx
    shard, cache, dev, sync, best, dirs, res, tm = w.shard, w.cache, w.dev, w.sync, w.best, w.dirs, w.res, w.tm

    mod = TRACKS[rank]
    t = time.time()
    root = out / "yolo_data" / mod.TASK
    Y.export(cache, mod.yolo_labels, mod.NAMES, root, cfg.smoke_train_images if smoke else COUNTS["train"])
    weights = Y.train(cfg, root, out / "yolo" / mod.TASK, cfg.smoke_yolo_hours if smoke else cfg.yolo_hours)
    tm["yolo_train_hours"] = (time.time() - t) / 3600
    log(f"YOLO {mod.TASK}: {weights}, {tm['yolo_train_hours']:.2f} h")
    torch.cuda.empty_cache()
    sync.barrier("yolo")

    plant_crop_stage(w)

    lmodel = load_eomt(best / "leaf_eomt", cfg.leaf_size).to(dev).eval()
    for s in ("val", "test"):
        tm[f"leaf_{s}"] = per_image(leaf_pass(cfg, lmodel, cache, s, shard[s], dirs, dev, log))
    del lmodel
    sem = SM.load_sem("m2f", best / "sem_m2f", None).to(dev).eval()
    with torch.no_grad():
        for s in ("val", "test"):
            tm[f"sem_{s}"] = per_image(sem_pass(cfg, sem, cache, s, shard[s], dirs, dev))
    del sem
    torch.cuda.empty_cache()
    from ultralytics import YOLO
    yolos = {m.TASK: YOLO(str(out / "yolo" / m.TASK / "run" / "weights" / "last.pt")) for m in TRACKS}
    for s in ("val", "test"):
        tm[f"yolo_{s}"] = per_image(yolo_pass(cfg, yolos, cache, s, shard[s], dirs))
    del yolos
    sync.barrier("inference")

    tracks = {m.TASK: S.Track(cfg, m, dirs, cache, names["val"]) for m in TRACKS}
    t = time.time()
    dec = {k: sync.decide(k, lambda tr=tr, k=k: S.decide(tr, idx["val"], out / f"rescore_{k}.pkl", log))
           for k, tr in tracks.items()}
    tm["decide_s"] = time.time() - t
    times = []
    for k, tr in tracks.items():
        models = pickle.loads((out / f"rescore_{k}.pkl").read_bytes())
        for split, sub in (("val", "val_pred"), ("test", "pred")):
            for i in shard[split]:
                t = time.time()
                M.write_txt(out / sub / f"{k}_bboxes" / (Path(names[split][i]).stem + ".txt"),
                            S.final(tr, dec[k], models, split, i))
                times.append(time.time() - t)
    tm["files"] = per_image(times)
    res.update({"decision": dec, "timing": tm})
    (out / f"result_{rank}.json").write_text(json.dumps(res))
    log(f"worker finished: {json.dumps({'decision': {k: (d['key'], d['ap']) for k, d in dec.items()}, 'timing': tm})}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
