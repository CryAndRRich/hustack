import itertools
import sys
import time

import numpy as np
import torch
from PIL import Image

from common import plant
from common import sem_model as M
from common.runtime import RANKS, Worker, per_image
from common.sem_metric import MERGE, ious
from common.stages import crop_map, load_map, plant_crop_stage
from common.train import train
from semantic.config import Config

MODELS = ["a", "b", "c"]


def subsets(xs):
    return ["".join(c) for k in range(1, len(xs) + 1) for c in itertools.combinations(xs, k)]


def train_sem(cfg, smoke, work, best, mdir, cache, qidx, dev, log, rank):
    spec = cfg.sem_train[rank]
    model = M.load_sem(spec["kind"], best / spec["init"], spec["size"]).to(dev)
    qe = {}

    def cb(name, m):
        qe[name] = M.quick_iou(m, cache, qidx, *spec["quick"], cfg.tile_batch, dev)
        log(f"quick val semantic {spec['name']} weed IoU {name}: {qe[name]:.2f}")

    log(f"semantic model {spec}: {cfg.train_hours:.2f} h")
    cb("start", model)
    kind = spec["kind"]
    lr, mem = (cfg.lr_plant, cfg.mem_limit_m2f) if kind == "m2f" else (cfg.lr_eomt, cfg.mem_limit_eomt)
    stats = train(cfg, model, kind, M.SemData(cfg, work / "cache", spec), 1, cfg.accum, cfg.train_hours, lr, mem, dev,
                  smoke, log, on_cycle=cb, semantic=True)
    cb("SWA", model)
    model.save_pretrained(mdir / f"sem_{spec['name']}")
    del model
    torch.cuda.empty_cache()
    return {**stats, "quick_val": qe}


def candidates(cfg):
    return [(m, v, f, o) for m in subsets(MODELS) for v in subsets(list(cfg.sem_views)) for f in cfg.weed_factors
            for o in (False, True)]


def cand_name(c):
    return f"{c[0]}_{c[1]}_w{c[2]}" + ("_crop" if c[3] else "")


def probs(cfg, sems, x, pairs):
    return {(m, v): M.view(sems[m], x, *cfg.sem_views[v], cfg.tile_batch) for m, v in pairs}


def source(pv, ms, vs):
    return torch.stack([pv[(m, v)].float() for m in ms for v in vs]).mean(0)


def labels(p, f, crop):
    lab = (p * torch.tensor([1.0, 1.0, f], device=p.device)[:, None, None]).argmax(0)
    return torch.where(crop, 1, lab) if crop is not None else lab


def crop_mask(cfg, dirs, split, i, dev):
    return torch.from_numpy(crop_map(cfg, *load_map(dirs, split, i), cfg.crop_decision) > 0).to(dev)


def sem_val(cfg, sems, cache, shard, dirs, cands, dev, log):
    conf, times = np.zeros((len(cands), 9), np.int64), []
    pairs = [(m, v) for m in MODELS for v in cfg.sem_views]
    for k, i in enumerate(shard):
        t = time.time()
        crop = crop_mask(cfg, dirs, "val", i, dev)
        pv = probs(cfg, sems, plant.image_tensor(cache["img_val"][i], dev), pairs)
        gt = torch.from_numpy(MERGE[np.asarray(cache["sem_val"][i]).astype(np.int64)]).to(dev).ravel() * 3
        src = {}
        for j, (ms, vs, f, o) in enumerate(cands):
            if (ms, vs) not in src:
                src = {(ms, vs): source(pv, ms, vs)}
            lab = labels(src[(ms, vs)], f, crop if o else None)
            conf[j] += torch.bincount(gt + lab.ravel(), minlength=9).cpu().numpy()
        times.append(time.time() - t)
        if k % 50 == 0:
            log(f"semantic val {k}/{len(shard)} {np.sum(times):.0f} s")
    return conf, times


def choose_sem(paths, cands, log):
    conf = sum(np.load(p)["conf"] for p in paths)
    rows = [(cand_name(c), *ious(conf[k].reshape(3, 3))) for k, c in enumerate(cands)]
    order = sorted(range(len(rows)), key=lambda k: -sum(rows[k][1:]))
    table = {rows[k][0]: {"soil": rows[k][1], "crop": rows[k][2], "weed": rows[k][3]} for k in order}
    show = set(order[:10])
    for key in (lambda c: c[0], lambda c: c[1], lambda c: c[3]):
        best = {}
        for k in order:
            best.setdefault(key(cands[k]), k)
        show |= set(best.values())
    show |= {k for k, c in enumerate(cands) if c == ("a", "gt", 1.0, True)}
    for k in sorted(show, key=order.index):
        r = rows[k]
        log(f"semantics {r[0]}: mIoU {np.mean(r[1:]):.2f} (soil {r[1]:.2f} crop {r[2]:.2f} weed {r[3]:.2f})")
    b = order[0]
    chosen = {"candidate": list(cands[b]), "name": rows[b][0], "miou": float(np.mean(rows[b][1:])),
              "iou_soil": rows[b][1], "iou_crop": rows[b][2], "iou_weed": rows[b][3]}
    log(f"semantics chosen: {chosen}")
    return {"chosen": chosen, "table": table}


def write_test(cfg, sems, cache, shard, dirs, c, names, pred, dev):
    ms, vs, f, o = c
    times = []
    for i in shard:
        t = time.time()
        pv = probs(cfg, sems, plant.image_tensor(cache["img_test"][i], dev), [(m, v) for m in ms for v in vs])
        lab = labels(source(pv, ms, vs), f, crop_mask(cfg, dirs, "test", i, dev) if o else None)
        Image.fromarray(lab.cpu().numpy().astype(np.uint8)).save(pred / names[i])
        times.append(time.time() - t)
    return times


DIRS = ("plant_val", "plant_test", "maps_val", "maps_test", "crop_val", "crop_test", "sem_val", "pred/semantics", "models")


def main(rank, mode, cfg_path):
    w = Worker(Config, rank, mode, cfg_path, DIRS)
    rank, cfg, smoke, log, work, names = w.rank, w.cfg, w.smoke, w.log, w.work, w.names
    shard, cache, dev, sync, best, dirs, res, tm = w.shard, w.cache, w.dev, w.sync, w.best, w.dirs, w.res, w.tm
    mdir, qidx = dirs["models"], w.quick_idx()

    res["train_sem"] = train_sem(cfg, smoke, work, best, mdir, cache, qidx, dev, log, rank)

    plant_crop_stage(w)
    sync.barrier("trained")

    sems = {"a": M.load_sem("m2f", best / "sem_m2f", None)}
    for spec in cfg.sem_train:
        sems[spec["name"]] = M.load_sem(spec["kind"], mdir / f"sem_{spec['name']}", cfg.sem_input)
    sems = {k: m.to(dev).eval() for k, m in sems.items()}
    cands = candidates(cfg)
    conf, t = sem_val(cfg, sems, cache, shard["val"], dirs, cands, dev, log)
    np.savez(dirs["sem_val"] / f"conf_{rank}.npz", conf=conf)
    tm["sem_val"] = per_image(t)
    sync.barrier("sem")
    sv = sync.decide("sem", lambda: choose_sem([dirs["sem_val"] / f"conf_{r}.npz" for r in range(RANKS)], cands, log))
    c = sv["chosen"]["candidate"]
    tm["sem_test"] = per_image(write_test(cfg, sems, cache, shard["test"], dirs, c, names["test"], dirs["pred/semantics"], dev))
    res.update({"decision": sv["chosen"], "table": sv["table"], "timing": tm})
    w.finish(["decision", "timing"])


if __name__ == "__main__":
    main(*sys.argv[1:4])
