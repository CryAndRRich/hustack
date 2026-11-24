import sys
import time

import numpy as np
import torch
from PIL import Image

from common import cropcls, leaf, plant, semantic
from common.models import load_m2f
from common.pq import instance_stats
from common.runtime import RANKS, Worker, per_image
from common.sem_metric import MERGE, crop_gt, ious
from common.stages import crop_curves, crop_leaf_stage, crop_map, load_crop_cls, load_map, plant_stage
from common.train import train
from hierarchical_panoptic.config import Config

SOURCES = [f"{m}_{v}" for m in [f"m{r}" for r in range(RANKS)] + ["ens"] for v in ("g", "t", "gt")]


def train_sem(cfg, smoke, work, best, mdir, cache, qidx, dev, log, rank):
    model = semantic.load_init(best / cfg.sem_init[rank]).to(dev)
    qe = {}

    def cb(name, m):
        qe[name] = semantic.quick_iou(cfg, m, cache, qidx, dev)
        log(f"quick val semantic {rank} weed IoU {name}: {qe[name]:.2f}")

    log(f"semantic model {rank}: from {cfg.sem_init[rank]}, zoom {cfg.sem_zoom_min}-{cfg.sem_zoom_max}, "
        f"weed windows {cfg.sem_weed_p}, {cfg.train_hours:.2f} h")
    stats = train(cfg, model, "m2f", semantic.SemSet(cfg, work / "cache"), 1, cfg.accum, cfg.train_hours, cfg.lr_plant,
                  cfg.mem_limit_m2f, dev, smoke, log, on_cycle=cb, semantic=True)
    cb("SWA", model)
    model.save_pretrained(mdir / f"sem_{rank}")
    del model
    torch.cuda.empty_cache()
    return {**stats, "quick_val": qe}


def crop_pass(cfg, cls, cache, split, shard, maps_dir, crop_dir, lmaps_dir, gate, dev, cur=None):
    times = []
    for k, i in enumerate(shard):
        t = time.time()
        pm, ps = leaf.plant_map(maps_dir, i)
        pc = cropcls.crop_probs(cfg, cls, np.asarray(cache[f"img_{split}"][i]), pm, dev)
        np.save(crop_dir / f"{i}.npy", pc)
        lm = np.where(np.concatenate([[0.0], ps])[pm] >= gate, pm, 0)
        np.savez_compressed(lmaps_dir / f"{i}.npz", map=lm.astype(np.uint16), scores=ps)
        if cur is not None:
            vis = np.asarray(cache["pvis_val"][i])
            gtc = crop_gt(np.asarray(cache["plant_val"][i]), np.asarray(cache["sem_val"][i]), vis)
            cur.add(k, lambda pm=pm, ps=ps, pc=pc, gtc=gtc, vis=vis: crop_curves(cfg, pm, ps, pc, gtc, vis))
        times.append(time.time() - t)
    return times


def candidates(cfg):
    ws = [None] + cfg.weed_scores
    return [(None, None, s) for s in ws] + [(src, f, s) for src in SOURCES for f in cfg.weed_factors for s in ws]


def cand_name(c):
    src, f, s = c
    return "none" if src is None and s is None else f"{src or 'inst'}" + (f"_w{f}" if src else "") + (f"_s{s}" if s is not None else "")


def needed(c):
    if c[0] is None:
        return [], []
    m, v = c[0].split("_")
    return list(range(RANKS)) if m == "ens" else [int(m[1:])], ["g", "t"] if v == "gt" else [v]


def sources(sems, x, cfg, ranks, vs):
    pv = {r: semantic.views(cfg, sems[r], x, tiles="t" in vs) for r in ranks}
    out = {}
    for src in SOURCES:
        rs, vv = needed((src, None, None))
        if set(rs) <= set(ranks) and set(vv) <= set(vs):
            out[src] = torch.stack([pv[r][v].float() for r in rs for v in vv]).mean(0)
    return out


def labels(c, srcs, crop, weed_maps):
    src, f, s = c
    if src is None:
        lab = torch.zeros(crop.shape, dtype=torch.long, device=crop.device)
    else:
        lab = (srcs[src] * torch.tensor([1.0, 1.0, f], device=crop.device)[:, None, None]).argmax(0)
    if s is not None:
        lab = torch.where(weed_maps[s], 2, lab)
    return torch.where(crop, 1, lab)


def weed_maps(cfg, pm, ps, pc, ct, dev):
    pmt = torch.from_numpy(pm.astype(np.int64)).to(dev)
    return {s: torch.from_numpy(np.concatenate([[False], (pc < ct) & (ps >= s)])).to(dev)[pmt] for s in cfg.weed_scores}


def sem_val(cfg, sems, cache, shard, dirs, dec, cands, dev, log):
    conf, times = np.zeros((len(cands), 9), np.int64), []
    for k, i in enumerate(shard):
        t = time.time()
        pm, ps, pc = load_map(dirs, "val", i)
        crop = torch.from_numpy(crop_map(cfg, pm, ps, pc, dec) > 0).to(dev)
        wm = weed_maps(cfg, pm, ps, pc, dec["crop_threshold"], dev)
        srcs = sources(sems, plant.image_tensor(cache["img_val"][i], dev), cfg, list(range(RANKS)), ["g", "t"])
        gt = torch.from_numpy(MERGE[np.asarray(cache["sem_val"][i]).astype(np.int64)]).to(dev).ravel() * 3
        conf += torch.stack([torch.bincount(gt + labels(c, srcs, crop, wm).ravel(), minlength=9) for c in cands]).cpu().numpy()
        times.append(time.time() - t)
        if k % 50 == 0:
            log(f"semantic val {k}/{len(shard)} {np.sum(times):.0f} s")
    return conf, times


def choose_sem(paths, cands, log):
    conf = sum(np.load(p)["conf"] for p in paths)
    rows = [(cand_name(c), *ious(conf[k].reshape(3, 3))) for k, c in enumerate(cands)]
    order = sorted(range(len(rows)), key=lambda k: -(rows[k][1] + rows[k][3]))
    table = {rows[k][0]: {"soil": rows[k][1], "crop": rows[k][2], "weed": rows[k][3]} for k in order}
    best_src = {}
    for k in order:
        best_src.setdefault(cands[k][0], k)
    for k in sorted(set(order[:10]) | set(best_src.values()), key=order.index):
        log(f"semantics {rows[k][0]}: IoU soil {rows[k][1]:.2f} crop {rows[k][2]:.2f} weed {rows[k][3]:.2f}")
    b = order[0]
    chosen = {"candidate": list(cands[b]), "name": rows[b][0], "iou_soil": rows[b][1], "iou_weed": rows[b][3]}
    log(f"semantics chosen: {chosen}")
    return {"chosen": chosen, "table": table}


def write_test(cfg, sems, cache, shard, dirs, dec, c, names, pred, dev):
    ranks, vs = needed(c)
    times = []
    for i in shard:
        t = time.time()
        pm, ps, pc = load_map(dirs, "test", i)
        cm = crop_map(cfg, pm, ps, pc, dec)
        srcs = sources(sems, plant.image_tensor(cache["img_test"][i], dev), cfg, ranks, vs) if ranks else {}
        lab = labels(c, srcs, torch.from_numpy(cm > 0).to(dev), weed_maps(cfg, pm, ps, pc, dec["crop_threshold"], dev))
        Image.fromarray(lab.cpu().numpy().astype(np.uint8)).save(pred / "semantics" / names[i])
        Image.fromarray(cm.astype(np.uint16)).save(pred / "plant_instances" / names[i])
        times.append(time.time() - t)
    return times


DIRS = ("plant_val", "plant_test", "maps_val", "maps_test", "crop_val", "crop_test", "lmaps_val", "lmaps_test", "sem_val",
        "leaf_val", "leaf_test", "pred/semantics", "pred/plant_instances", "pred/leaf_instances", "val_pred/plant_instances",
        "val_pred/leaf_instances", "models")


def main(rank, mode, cfg_path):
    w = Worker(Config, rank, mode, cfg_path, DIRS)
    rank, cfg, smoke, log, work, out, names = w.rank, w.cfg, w.smoke, w.log, w.work, w.out, w.names
    shard, cache, dev, sync, best, dirs, res, tm = w.shard, w.cache, w.dev, w.sync, w.best, w.dirs, w.res, w.tm
    pos, grp = w.groups()
    mdir, qidx = dirs["models"], w.quick_idx()
    gates = sorted(set(cfg.leaf_gates))

    res["train_sem"] = train_sem(cfg, smoke, work, best, mdir, cache, qidx, dev, log, rank)
    sync.barrier("trained")

    plant_stage(w, cfg.min_score)

    cls = load_crop_cls(cfg, best / "crop_cls", dev)
    pd, nv = dirs["plant_val"], len(shard["val"])
    cur = plant.Curves(nv)
    tm["crop_val"] = per_image(crop_pass(cfg, cls, cache, "val", shard["val"], dirs["maps_val"], dirs["crop_val"],
                                         dirs["lmaps_val"], gates[0], dev, cur))
    cur.save(pd / f"c1_{rank}.npz", shard["val"])
    tm["crop_test"] = per_image(crop_pass(cfg, cls, cache, "test", shard["test"], dirs["maps_test"], dirs["crop_test"],
                                          dirs["lmaps_test"], gates[0], dev))
    del cls
    torch.cuda.empty_cache()
    sync.barrier("plant_c1")
    c1 = sync.decide("plant_c1", lambda: plant.choose([pd / f"c1_{r}.npz" for r in range(RANKS)], cfg, log, "crop plants"))
    ch = c1["chosen"]
    dec_plant = {"crop_threshold": float(ch["candidate"][1:]), "threshold": ch["threshold"], "offset": ch["offset"],
                 "val_pq": ch["val_pq"]}
    log(f"crop plant decision: {dec_plant}")
    rows, fps = [], []
    for i in shard["val"]:
        cm = crop_map(cfg, *load_map(dirs, "val", i), dec_plant)
        Image.fromarray(cm.astype(np.uint16)).save(dirs["val_pred/plant_instances"] / names["val"][i])
        vis = np.asarray(cache["pvis_val"][i])
        r, f = instance_stats(cm, crop_gt(np.asarray(cache["plant_val"][i]), np.asarray(cache["sem_val"][i]), vis), vis, None)
        rows += [(i, *row) for row in r]
        fps += f
    res["diag_plant"] = {"rows": rows, "fp": fps}

    sems = [load_m2f(mdir / f"sem_{r}").to(dev).eval() for r in range(RANKS)]
    cands = candidates(cfg)
    conf, t = sem_val(cfg, sems, cache, shard["val"], dirs, dec_plant, cands, dev, log)
    np.savez(dirs["sem_val"] / f"conf_{rank}.npz", conf=conf)
    tm["sem_val"] = per_image(t)
    sync.barrier("sem")
    sv = sync.decide("sem", lambda: choose_sem([dirs["sem_val"] / f"conf_{r}.npz" for r in range(RANKS)], cands, log))
    c = tuple(sv["chosen"]["candidate"])
    tm["sem_test"] = per_image(write_test(cfg, sems, cache, shard["test"], dirs, dec_plant, c, names["test"],
                                          out / "pred", dev))
    del sems
    torch.cuda.empty_cache()

    dec_leaf, leaf_tables, counts = crop_leaf_stage(w, "lmaps", gates, pos, grp)
    res.update({"leaf_test_instances": counts, "decision": {"plant": dec_plant, "semantics": sv["chosen"], "leaf": dec_leaf},
                "tables": {"plant": c1["table"], "semantics": sv["table"], "leaf": leaf_tables}, "timing": tm})
    w.finish(["decision", "timing"])


if __name__ == "__main__":
    main(*sys.argv[1:4])
