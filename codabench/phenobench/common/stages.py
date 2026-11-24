import time
from functools import lru_cache

import numpy as np

from common import cropcls, leaf, plant
from common import items as I
from common.models import load_eomt, load_m2f
from common.pq import pq_curve
from common.runtime import RANKS, per_image


def val_gt(cache, i):
    return np.asarray(cache["leaf_val"][i]), np.asarray(cache["lvis_val"][i]), None


def crop_filter(gate, t, pc):
    return lambda its: [it for it in its if it[4][0] >= gate and pc(it[4][1][0])[it[4][1][1] - 1] >= t]


def plant_maps(cfg, models, cache, split, shard, d, maps_dir, gate_min, dev, log):
    times = []
    for k, i in enumerate(shard):
        t = time.time()
        src = plant.sources(models, plant.image_tensor(cache[f"img_{split}"][i], dev), cfg, [cfg.plant_candidate])
        I.save(plant.candidate(src, cfg.plant_candidate, cfg), d / f"{i}.pkl")
        times.append(time.time() - t)
        if k % 50 == 0:
            log(f"plant {split} {k}/{len(shard)} {np.sum(times):.0f} s")
    plant.write_maps(cfg, shard, lambda i: I.load(d / f"{i}.pkl"), cfg.plant_threshold, cfg.plant_offset, gate_min, maps_dir)
    return times


def plant_stage(w, gate_min):
    import torch
    models = {"v3": load_m2f(w.best / "plant_m2f").to(w.dev).eval(), "m": load_m2f(w.best / "plant_zoom_m2f").to(w.dev).eval()}
    for s in ("val", "test"):
        w.tm[f"plant_{s}"] = per_image(plant_maps(w.cfg, models, w.cache, s, w.shard[s], w.dirs[f"plant_{s}"],
                                                  w.dirs[f"maps_{s}"], gate_min, w.dev, w.log))
    del models
    torch.cuda.empty_cache()


def load_crop_cls(cfg, folder, dev):
    import torch
    cls = []
    for r in range(RANKS):
        m = cropcls.build(cfg)
        m.load_state_dict(torch.load(folder / f"rank{r}.pt", map_location="cpu"))
        cls.append(m.to(dev).eval())
    return cls


def crop_stage(w, cls):
    for s in ("val", "test"):
        w.tm[f"crop_{s}"] = per_image(crop_pass(w.cfg, cls, w.cache, s, w.shard[s], w.dirs[f"maps_{s}"], w.dirs[f"crop_{s}"], w.dev))


def plant_crop_stage(w):
    import torch
    plant_stage(w, w.cfg.min_score)
    crop_stage(w, load_crop_cls(w.cfg, w.best / "crop_cls", w.dev))
    torch.cuda.empty_cache()


def crop_pass(cfg, models, cache, split, shard, maps_dir, out_dir, dev):
    times = []
    for i in shard:
        t = time.time()
        pm, _ = leaf.plant_map(maps_dir, i)
        np.save(out_dir / f"{i}.npy", cropcls.crop_probs(cfg, models, np.asarray(cache[f"img_{split}"][i]), pm, dev))
        times.append(time.time() - t)
    return times


def leaf_val(cfg, model, cache, shard, maps_dir, crop_dir, d, gates, dev, log):
    cur, times = plant.Curves(len(shard)), []
    for k, i in enumerate(shard):
        t = time.time()
        pm, ps = leaf.plant_map(maps_dir, i)
        its = leaf.crop_items(cfg, model, np.asarray(cache["img_val"][i]), pm, ps, dev, i)
        I.save(its, d / f"{i}.pkl")
        pc = np.load(crop_dir / f"{i}.npy")
        gt, vis, _ = val_gt(cache, i)
        cur.add(k, lambda its=its, pc=pc, gt=gt, vis=vis: {
            f"g{g}_c{c}": plant.curve(crop_filter(g, c, lambda _: pc)(its), gt, vis, None, cfg)
            for g in gates for c in cfg.crop_thresholds})
        times.append(time.time() - t)
        if k % 50 == 0:
            log(f"leaf val {k}/{len(shard)} {np.sum(times):.0f} s")
    return cur, times


def load_map(dirs, split, i):
    pm, ps = leaf.plant_map(dirs[f"maps_{split}"], i)
    return pm, ps, np.load(dirs[f"crop_{split}"] / f"{i}.npy")


def crop_curves(cfg, pm, ps, pc, gtc, vis):
    return {f"c{c}": pq_curve(pm, np.where(pc >= c, ps, -1.0), gtc, vis, None, cfg.thresholds, cfg.small_offsets,
                              cfg.size_split) for c in cfg.crop_thresholds}


def crop_map(cfg, pm, ps, pc, dec):
    ok = I.size_keep(pm, ps, dec["threshold"], dec["offset"], cfg.size_split) & (pc >= dec["crop_threshold"])
    return np.where(np.concatenate([[False], ok])[pm], pm, 0)


def crop_leaf_stage(w, maps, gates, pos, grp):
    import torch
    cfg, cache, shard, dirs, dev, log, sync, phase = w.cfg, w.cache, w.shard, w.dirs, w.dev, w.log, w.sync, w.phase
    lmodel = load_eomt(w.best / "leaf_eomt", cfg.leaf_size).to(dev).eval()
    ld, nv = dirs["leaf_val"], len(shard["val"])
    cur, t = leaf_val(cfg, lmodel, cache, shard["val"], dirs[f"{maps}_val"], dirs["crop_val"], ld, gates, dev, log)
    cur.save(ld / f"c1_{w.rank}.npz", shard["val"])
    w.tm["leaf_val"] = per_image(t)
    sync.barrier("leaf_c1")
    l1 = sync.decide("leaf_c1", lambda: plant.choose([ld / f"c1_{r}.npz" for r in range(RANKS)], cfg, log, "leaf gate/crop"))
    lgate, cthr = (float(v) for v in l1["chosen"]["candidate"][1:].split("_c"))
    post = crop_filter(lgate, cthr, lru_cache(maxsize=64)(lambda j: np.load(dirs["crop_val"] / f"{j}.npy")))
    lload = lru_cache(maxsize=64)(lambda j: I.load(ld / f"{j}.pkl"))
    cur = phase("leaf_stitch", nv, lambda: plant.stitch_pass(cfg, lambda i: val_gt(cache, i), shard["val"], lload,
                                                             pos["val"], grp["val"], post)[0])
    cur.save(ld / f"c2_{w.rank}.npz", shard["val"])
    sync.barrier("leaf_c2")
    l2 = sync.decide("leaf_c2", lambda: plant.choose([ld / f"c2_{r}.npz" for r in range(RANKS)], cfg, log, "leaf stitch"))
    lstitch, lthr, loff = l2["chosen"]["candidate"] == "stitch1", l2["chosen"]["threshold"], l2["chosen"]["offset"]
    dec = {"gate": lgate, "crop_threshold": cthr, "stitch": lstitch, "threshold": lthr, "offset": loff,
           "val_pq": l2["chosen"]["val_pq"]}
    log(f"leaf decision: {dec}")
    rows, fps, _ = phase("leaf_val_maps", nv, lambda: leaf.write(
        cfg, shard["val"], lambda i: post(plant.chosen_items(i, lload, lstitch, pos["val"], grp["val"], cfg)), lgate, lthr,
        loff, dirs["val_pred/leaf_instances"], w.names["val"], gt_fn=lambda i: val_gt(cache, i)))
    w.res["diag_leaf"] = {"rows": rows, "fp": fps}
    w.tm["leaf_test"] = per_image(leaf.test_items(cfg, lmodel, cache, shard["test"], dirs[f"{maps}_test"], dirs["leaf_test"],
                                                  dev, log))
    del lmodel
    torch.cuda.empty_cache()
    sync.barrier("leaf_test")
    post_t = crop_filter(lgate, cthr, lru_cache(maxsize=64)(lambda j: np.load(dirs["crop_test"] / f"{j}.npy")))
    ltload = lru_cache(maxsize=64)(lambda j: I.load(dirs["leaf_test"] / f"{j}.pkl"))
    _, _, counts = phase("leaf_test_maps", len(shard["test"]), lambda: leaf.write(
        cfg, shard["test"], lambda i: post_t(plant.chosen_items(i, ltload, lstitch, pos["test"], grp["test"], cfg)),
        lgate, lthr, loff, dirs["pred/leaf_instances"], w.names["test"]))
    return dec, [l1["table"], l2["table"]], counts
