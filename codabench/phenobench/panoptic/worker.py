import sys
import time

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

from common import plant
from common import items as I
from common import sem_model as M
from common.pq import instance_stats
from common.runtime import RANKS, Worker, per_image
from common.sem_metric import confusion
from common.stages import crop_curves, crop_map, load_map, plant_crop_stage
from panoptic.config import Config
from panoptic.metric import class_gt, pq_curves

EIGHT = np.ones((3, 3), bool)


def weed_candidates(cfg):
    return {f"w{c}_r{r}_t{int(t)}": (c, r, t) for r in cfg.recover_areas for t in (False, True)
            for c in cfg.weed_crop_thresholds}


def weed_base(pm, n, lab, r, trim):
    m, nr = pm, 0
    if r is not None:
        cc, k = ndimage.label((lab == 2) & (pm == 0), EIGHT)
        keep = np.bincount(cc.ravel(), minlength=k + 1) >= r
        keep[0] = False
        nr = int(keep.sum())
        new = np.zeros(k + 1, np.int64)
        new[keep] = n + 1 + np.arange(nr)
        m = np.where(new[cc] > 0, new[cc], pm)
    return (np.where(lab == 0, 0, m) if trim else m), nr


def weed_scores(ps, pc, c, nr):
    return np.concatenate([np.where(pc < c, ps, -1.0), np.full(nr, 2.0)])


def weed_curves(cfg, pm, ps, pc, lab, gtw, vis):
    out = {}
    for r in cfg.recover_areas:
        for trim in (False, True):
            m, nr = weed_base(pm, len(ps), lab, r, trim)
            sets = {f"w{c}_r{r}_t{int(trim)}": weed_scores(ps, pc, c, nr) for c in cfg.weed_crop_thresholds}
            out.update(pq_curves(m, sets, gtw, vis, cfg.thresholds, cfg.small_offsets, cfg.size_split))
    return out


def weed_map(cfg, pm, ps, pc, lab, wd):
    m, nr = weed_base(pm, len(ps), lab, wd["recover"], wd["trim"])
    ok = I.size_keep(m, weed_scores(ps, pc, wd["crop_threshold"], nr), wd["threshold"], wd["offset"], cfg.size_split)
    return np.where(np.concatenate([[False], ok])[m], m, 0)


def panoptic(cfg, pm, ps, pc, lab, dc, wd):
    cm = crop_map(cfg, pm, ps, pc, dc)
    wm = weed_map(cfg, pm, ps, pc, lab, wd)
    return np.where(cm > 0, cm, wm), np.where(cm > 0, 1, np.where(wm > 0, 2, lab)), wm


def sem_pass(cfg, model, cache, split, shard, dirs, dev, log, cur=None):
    times = []
    for k, i in enumerate(shard):
        t = time.time()
        x = plant.image_tensor(cache[f"img_{split}"][i], dev)
        p = torch.stack([M.view(model, x, *cfg.sem_views[v], cfg.tile_batch).float() for v in cfg.sem_views]).mean(0)
        lab = p.argmax(0).cpu().numpy().astype(np.uint8)
        np.savez_compressed(dirs[f"sem_{split}"] / f"{i}.npz", lab=lab)
        if cur is not None:
            pm, ps, pc = load_map(dirs, "val", i)
            plant_gt, sem_gt, vis = (np.asarray(cache[f"{c}_val"][i]) for c in ("plant", "sem", "pvis"))
            gtc, gtw = class_gt(plant_gt, sem_gt, vis, 1), class_gt(plant_gt, sem_gt, vis, 2)
            cur[0].add(k, lambda pm=pm, ps=ps, pc=pc, gtc=gtc, vis=vis: crop_curves(cfg, pm, ps, pc, gtc, vis))
            cur[1].add(k, lambda pm=pm, ps=ps, pc=pc, lab=lab, gtw=gtw, vis=vis: weed_curves(cfg, pm, ps, pc, lab, gtw, vis))
        times.append(time.time() - t)
        if k % 50 == 0:
            log(f"semantics {split} {k}/{len(shard)} {np.sum(times):.0f} s")
    return times


def write_maps(cfg, cache, split, shard, dirs, dc, wd, names, pred):
    conf, rows, fps, times = np.zeros((3, 3), np.int64), [], [], []
    for i in shard:
        t = time.time()
        pm, ps, pc = load_map(dirs, split, i)
        inst, sem, wm = panoptic(cfg, pm, ps, pc, np.load(dirs[f"sem_{split}"] / f"{i}.npz")["lab"], dc, wd)
        Image.fromarray(sem.astype(np.uint8)).save(pred / "semantics" / names[i])
        Image.fromarray(inst.astype(np.uint16)).save(pred / "plant_instances" / names[i])
        if split == "val":
            plant_gt, sem_gt, vis = (np.asarray(cache[f"{c}_val"][i]) for c in ("plant", "sem", "pvis"))
            conf += confusion(sem_gt, sem)
            r, f = instance_stats(wm, class_gt(plant_gt, sem_gt, vis, 2), vis, None)
            rows += [(i, *row) for row in r]
            fps += f
        times.append(time.time() - t)
    return times, {"conf": conf.tolist(), "rows": rows, "fp": fps}


DIRS = ("plant_val", "plant_test", "maps_val", "maps_test", "crop_val", "crop_test", "sem_val", "sem_test", "pred/semantics",
        "pred/plant_instances", "val_pred/semantics", "val_pred/plant_instances")


def main(rank, mode, cfg_path):
    w = Worker(Config, rank, mode, cfg_path, DIRS)
    rank, cfg, log, out, names = w.rank, w.cfg, w.log, w.out, w.names
    shard, cache, dev, sync, best, dirs, res, tm = w.shard, w.cache, w.dev, w.sync, w.best, w.dirs, w.res, w.tm

    plant_crop_stage(w)

    sem = M.load_sem("m2f", best / "sem_m2f", None).to(dev).eval()
    cur = (plant.Curves(len(shard["val"])), plant.Curves(len(shard["val"])))
    tm["sem_val"] = per_image(sem_pass(cfg, sem, cache, "val", shard["val"], dirs, dev, log, cur))
    pd = dirs["plant_val"]
    cur[0].save(pd / f"curves_crop_{rank}.npz", shard["val"])
    cur[1].save(pd / f"curves_weed_{rank}.npz", shard["val"])
    tm["sem_test"] = per_image(sem_pass(cfg, sem, cache, "test", shard["test"], dirs, dev, log))
    del sem
    torch.cuda.empty_cache()
    sync.barrier("curves")

    c1 = sync.decide("crop", lambda: plant.choose([pd / f"curves_crop_{r}.npz" for r in range(RANKS)], cfg, log, "crop plants"))
    ch = c1["chosen"]
    dc = {"crop_threshold": float(ch["candidate"][1:]), "threshold": ch["threshold"], "offset": ch["offset"],
          "val_pq": ch["val_pq"]}
    log(f"crop decision: {dc}")
    wc = weed_candidates(cfg)
    allowed = [k for k, v in wc.items() if v[0] <= max(dc["crop_threshold"], min(cfg.weed_crop_thresholds))]
    w1 = sync.decide("weed", lambda: plant.choose([pd / f"curves_weed_{r}.npz" for r in range(RANKS)], cfg, log, "weeds", allowed))
    ch = w1["chosen"]
    c, r, trim = wc[ch["candidate"]]
    wd = {"key": ch["candidate"], "crop_threshold": min(c, dc["crop_threshold"]), "recover": r, "trim": trim, "threshold": ch["threshold"],
          "offset": ch["offset"], "val_pq": ch["val_pq"]}
    log(f"weed decision: {wd}")

    t, res["val"] = write_maps(cfg, cache, "val", shard["val"], dirs, dc, wd, names["val"], out / "val_pred")
    tm["maps_val"] = per_image(t)
    tm["maps_test"] = per_image(write_maps(cfg, cache, "test", shard["test"], dirs, dc, wd, names["test"], out / "pred")[0])
    res.update({"decision": {"crop": dc, "weed": wd}, "tables": {"crop": c1["table"], "weed": w1["table"]}, "timing": tm})
    w.finish(["decision", "timing"])


if __name__ == "__main__":
    main(*sys.argv[1:4])
