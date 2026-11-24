import sys
from functools import lru_cache

import numpy as np
import torch

from common import items as I
from common import leaf, plant
from common.data import ZoomSet
from common.models import load_eomt, load_m2f
from common.runtime import RANKS, Worker, per_image
from common.train import train
from hierarchical_plant_leaf.config import Config


def quick(log, fn, store, tag):
    def cb(name, model):
        store[name] = fn(model)
        log(f"quick val {tag} PQ {name}: {store[name]:.2f}")
    return cb


ROLES = ("m2f", "eomt")


def train_zoom(cfg, smoke, work, best, mdir, cache, qidx, dev, log, rank):
    kind = ROLES[rank]
    v3 = load_m2f(best / "plant_m2f").to(dev)
    glob = plant.global_val_items(cfg, v3, cache, qidx, dev)
    del v3
    torch.cuda.empty_cache()
    ckpt = False
    if kind == "m2f":
        model = load_m2f(best / "plant_m2f").to(dev)
        size, stride, lr, mem = cfg.m2f_size, cfg.m2f_stride, cfg.lr_plant, cfg.mem_limit_m2f
    else:
        model = load_eomt(best / "leaf_eomt", cfg.eomt_size).to(dev)
        ckpt = bool(getattr(model, "supports_gradient_checkpointing", False))
        if ckpt:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        size, stride, lr, mem = cfg.eomt_size, cfg.eomt_stride, cfg.lr_eomt, cfg.mem_limit_eomt
    log(f"zoom specialist {kind}: input {size}, zoom {cfg.zoom_min}-{cfg.zoom_max}, label stride {stride}, "
        f"gradient checkpointing {ckpt}")
    qe = {}
    cb = quick(log, lambda m: plant.quick_eval(cfg, m, glob, cache, size // 2, size, dev), qe, f"zoom {kind}")
    cb("start", model)
    stats = train(cfg, model, kind, ZoomSet(cfg, work / "cache", size, stride), 1, cfg.accum, cfg.train_hours, lr, mem,
                  dev, smoke, log, on_cycle=cb)
    cb("SWA", model)
    model.save_pretrained(mdir / f"plant_zoom_{kind}")
    del model
    torch.cuda.empty_cache()
    return {**stats, "quick_val": qe}


DIRS = ("plant_val", "plant_val_maps", "plant_test", "plant_test_maps", "leaf_val", "leaf_test", "pred/plant_instances",
        "pred/leaf_instances", "val_pred/plant_instances", "val_pred/leaf_instances")


def main(rank, mode, cfg_path):
    w = Worker(Config, rank, mode, cfg_path, DIRS)
    rank, cfg, smoke, log, work, out, names = w.rank, w.cfg, w.smoke, w.log, w.work, w.out, w.names
    shard, cache, dev, sync, best, dirs, res, tm, phase = w.shard, w.cache, w.dev, w.sync, w.best, w.dirs, w.res, w.tm, w.phase
    pos, grp = w.groups()
    mdir, qidx = out / "models", w.quick_idx()

    res[f"train_zoom_{ROLES[rank]}"] = train_zoom(cfg, smoke, work, best, mdir, cache, qidx, dev, log, rank)
    sync.barrier("trained")

    pd, nv = dirs["plant_val"], len(shard["val"])
    models = {"v3": load_m2f(best / "plant_m2f").to(dev).eval(), "m": load_m2f(mdir / "plant_zoom_m2f").to(dev).eval(),
              "e": load_eomt(mdir / "plant_zoom_eomt", plant.SIZE).to(dev).eval()}
    cur, t = plant.pass_val(cfg, models, cache, shard["val"], pd, dev, log)
    cur.save(pd / f"c1_{rank}.npz", shard["val"])
    tm["plant_val"] = per_image(t)
    sync.barrier("plant_c1")
    c1 = sync.decide("plant_c1", lambda: plant.choose([pd / f"c1_{r}.npz" for r in range(RANKS)], cfg, log, "plant"))
    ch = c1["chosen"]
    name, pthr, poff = ch["candidate"], ch["threshold"], ch["offset"]
    dec_plant = {"candidate": name, "threshold": pthr, "offset": poff, "val_pq": ch["val_pq"]}
    log(f"plant decision: {dec_plant}")
    gates = sorted(set(cfg.leaf_gates + [pthr]))
    load = plant.loader(pd, f"_{name}")
    rows, fps, _ = phase("plant_val_maps", nv, lambda: plant.write_maps(
        cfg, shard["val"], load, pthr, poff, gates[0], dirs["plant_val_maps"], dirs["val_pred/plant_instances"],
        names["val"], gt_fn=lambda i: (*plant.val_gt(cache, i), np.asarray(cache["sem_val"][i]))))
    res["diag_plant"] = {"rows": rows, "fp": fps}

    tm["plant_test"] = per_image(plant.test_items(cfg, models, cache, shard["test"], name, dirs["plant_test"], dev, log))
    del models
    torch.cuda.empty_cache()
    sync.barrier("plant_test")
    load_t = plant.loader(dirs["plant_test"], "")
    _, _, counts = phase("plant_test_maps", len(shard["test"]), lambda: plant.write_maps(
        cfg, shard["test"], load_t, pthr, poff, gates[0], dirs["plant_test_maps"], dirs["pred/plant_instances"],
        names["test"]))
    res["plant_test_instances"] = counts

    lmodel = load_eomt(best / "leaf_eomt", cfg.leaf_size).to(dev).eval()
    ld = dirs["leaf_val"]
    cur, t = leaf.val_pass(cfg, lmodel, cache, shard["val"], dirs["plant_val_maps"], ld, gates, dev, log)
    cur.save(ld / f"c1_{rank}.npz", shard["val"])
    tm["leaf_val"] = per_image(t)
    sync.barrier("leaf_c1")
    l1 = sync.decide("leaf_c1", lambda: plant.choose([ld / f"c1_{r}.npz" for r in range(RANKS)], cfg, log, "leaf gate"))
    lgate = float(l1["chosen"]["candidate"][1:])
    lload = lru_cache(maxsize=64)(lambda j: I.load(ld / f"{j}.pkl"))
    post = lambda its: leaf.gated(its, lgate)
    cur = phase("leaf_stitch", nv, lambda: plant.stitch_pass(cfg, lambda i: leaf.val_gt(cache, i), shard["val"], lload,
                                                             pos["val"], grp["val"], post)[0])
    cur.save(ld / f"c2_{rank}.npz", shard["val"])
    sync.barrier("leaf_c2")
    l2 = sync.decide("leaf_c2", lambda: plant.choose([ld / f"c2_{r}.npz" for r in range(RANKS)], cfg, log, "leaf stitch"))
    lstitch, lthr, loff = l2["chosen"]["candidate"] == "stitch1", l2["chosen"]["threshold"], l2["chosen"]["offset"]
    dec_leaf = {"gate": lgate, "stitch": lstitch, "threshold": lthr, "offset": loff, "val_pq": l2["chosen"]["val_pq"]}
    log(f"leaf decision: {dec_leaf}")
    rows, fps, _ = phase("leaf_val_maps", nv, lambda: leaf.write(
        cfg, shard["val"], lambda i: plant.chosen_items(i, lload, lstitch, pos["val"], grp["val"], cfg), lgate, lthr, loff,
        dirs["val_pred/leaf_instances"], names["val"], gt_fn=lambda i: leaf.val_gt(cache, i)))
    res["diag_leaf"] = {"rows": rows, "fp": fps}

    tm["leaf_test"] = per_image(leaf.test_items(cfg, lmodel, cache, shard["test"], dirs["plant_test_maps"],
                                                dirs["leaf_test"], dev, log))
    del lmodel
    torch.cuda.empty_cache()
    sync.barrier("leaf_test")
    ltload = lru_cache(maxsize=64)(lambda j: I.load(dirs["leaf_test"] / f"{j}.pkl"))
    _, _, counts = phase("leaf_test_maps", len(shard["test"]), lambda: leaf.write(
        cfg, shard["test"], lambda i: plant.chosen_items(i, ltload, lstitch, pos["test"], grp["test"], cfg),
        lgate, lthr, loff, dirs["pred/leaf_instances"], names["test"]))
    res.update({"leaf_test_instances": counts, "decision": {"plant": dec_plant, "leaf": dec_leaf},
                "tables": {"plant": [c1["table"]], "leaf": [l1["table"], l2["table"]]}, "timing": tm})
    w.finish(["decision", "timing"])


if __name__ == "__main__":
    main(*sys.argv[1:4])
