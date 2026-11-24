import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from common import items as I
from common.fuse import fuse
from common.models import predict_batch
from common.pq import instance_stats, pq_curve
from common.stitch import stitch

SIZE = 1024


def image_tensor(arr, dev):
    return torch.from_numpy(np.array(arr)).to(dev).permute(2, 0, 1)[None].float()


def global_items(model, x, cfg):
    xs = F.interpolate(x, scale_factor=cfg.plant_scale, mode="bilinear", align_corners=False)
    p, s = predict_batch(model, xs, cfg, SIZE)[0]
    return I.from_probs(p, s, cfg.min_score)


def tile_starts(h):
    n = -(-SIZE // h)
    return [round(k * (SIZE - h) / (n - 1)) for k in range(n)]


def zoom_sources(models, x, cfg, h, srcs, size=SIZE):
    starts = tile_starts(h)
    tiles = [(t, l) for t in starts for l in starts]
    run = sorted({s for s in srcs if s != "v"} | ({"m", "e"} if "v" in srcs else set()))
    out = {s: [] for s in srcs}
    for a in range(0, len(tiles), cfg.tile_batch):
        batch = tiles[a:a + cfg.tile_batch]
        xt = torch.cat([F.interpolate(x[:, :, t:t + h, l:l + h], size=(size, size), mode="bilinear", align_corners=False)
                        for t, l in batch])
        preds = {k: predict_batch(models[k], xt, cfg, h) for k in run}
        for j, (t, l) in enumerate(batch):
            for src in srcs:
                p, sc = fuse([preds["m"][j], preds["e"][j]], 2, cfg.fuse_iou) if src == "v" else preds[src][j]
                for it in I.from_probs(p, sc, cfg.min_score, t, l):
                    y0, x0, y1, x1 = I.box(it)
                    if not ((y0 == t and t > 0) or (x0 == l and l > 0) or (y1 == t + h and t + h < SIZE)
                            or (x1 == l + h and l + h < SIZE)):
                        out[src].append(it)
    return {s: I.by_score(v) for s, v in out.items()}


ZOOM_SRCS = ("m", "e", "v")


def candidate_names(cfg):
    return (["v3"] + [f"z2_{s}" for s in ("v3",) + ZOOM_SRCS]
            + [f"z3_{s}_{g}" for s in ZOOM_SRCS for g in cfg.z3_gates])


def needs(names):
    z2, z3 = set(), set()
    for n in names:
        parts = n.split("_")
        z2.add("v3" if n == "v3" else parts[1])
        if parts[0] == "z3":
            z3.add(parts[1])
    return sorted(z2), sorted(z3)


def sources(models, x, cfg, names):
    z2, z3 = needs(names)
    src = {"g": global_items(models["v3"], x, cfg)}
    src.update({f"z2:{k}": v for k, v in zoom_sources(models, x, cfg, SIZE // 2, z2).items()})
    if z3:
        src.update({f"z3:{k}": v for k, v in zoom_sources(models, x, cfg, -(-SIZE // 3), z3).items()})
    return src


def candidate(src, name, cfg):
    if name == "v3":
        return I.merge_zoom(src["g"], src["z2:v3"], cfg.z2_gate)
    kind, s, *gate = name.split("_")
    base = I.merge_zoom(src["g"], src[f"z2:{s}"], cfg.z2_gate, contain=True)
    return base if kind == "z2" else I.merge_zoom(base, src[f"z3:{s}"], int(gate[0]), contain=True)


def curve(its, gt, vis, dc, cfg):
    canvas, kept, _ = I.paint(its, cfg.keep_frac, cfg.min_score)
    return pq_curve(canvas, kept, gt, vis, dc, cfg.thresholds, cfg.small_offsets, cfg.size_split)


def choose(paths, cfg, log, tag, keys=None):
    parts = [np.load(p) for p in paths]
    table, best = {}, None
    for c in keys or [k for k in parts[0].files if k != "idx"]:
        m = np.nanmean(np.concatenate([z[c] for z in parts]), 0) * 100
        o, t = np.unravel_index(int(np.argmax(m)), m.shape)
        z = cfg.small_offsets.index(0.0)
        table[c] = {"threshold": cfg.thresholds[t], "offset": cfg.small_offsets[o], "val_pq": float(m[o, t]),
                    "val_pq_offset0": float(m[z].max())}
        log(f"{tag} {c}: thr {cfg.thresholds[t]} small offset {cfg.small_offsets[o]} PQ {m[o, t]:.2f} "
            f"(offset 0: {m[z].max():.2f})")
        if best is None or m[o, t] > best["val_pq"]:
            best = {"candidate": c, **table[c]}
    log(f"{tag} chosen: {best}")
    return {"chosen": best, "table": table}


class Curves:
    def __init__(self, n):
        self.pool, self.jobs, self.n = ThreadPoolExecutor(1), [], n

    def add(self, k, fn):
        self.jobs.append((k, self.pool.submit(fn)))
        while sum(not j.done() for _, j in self.jobs) > 2:
            time.sleep(0.01)

    def save(self, path, idx):
        rows = [(k, j.result()) for k, j in self.jobs]
        assert rows, f"no curves for {path}"
        keys = list(rows[0][1].keys())
        arr = {c: np.full((self.n,) + rows[0][1][c].shape, np.nan) for c in keys}
        for k, res in rows:
            for c in keys:
                arr[c][k] = res[c]
        np.savez(path, idx=np.array(idx), **arr)
        self.pool.shutdown()


def val_gt(cache, i):
    return np.asarray(cache["plant_val"][i]), np.asarray(cache["pvis_val"][i])


def pass_val(cfg, models, cache, shard, d, dev, log):
    names = candidate_names(cfg)
    cur, times = Curves(len(shard)), []
    for k, i in enumerate(shard):
        t = time.time()
        src = sources(models, image_tensor(cache["img_val"][i], dev), cfg, names)
        cands = {n: candidate(src, n, cfg) for n in names}
        for n, v in cands.items():
            I.save(v, d / f"final_{i}_{n}.pkl")
        gt, vis = val_gt(cache, i)
        cur.add(k, lambda cands=cands, gt=gt, vis=vis: {n: curve(v, gt, vis, None, cfg) for n, v in cands.items()})
        times.append(time.time() - t)
        if k % 50 == 0:
            log(f"plant val {k}/{len(shard)} {np.sum(times):.0f} s")
    return cur, times


def stitch_pass(cfg, gt_fn, shard, load, pos, groups, post=lambda its: its):
    cur, times = Curves(len(shard)), []
    for k, i in enumerate(shard):
        t = time.time()
        own, st = post(load(i)), post(stitch(i, load, pos, groups, cfg.stitch_iou))
        gt, vis, dc = gt_fn(i)
        cur.add(k, lambda own=own, st=st, gt=gt, vis=vis, dc=dc: {"stitch0": curve(own, gt, vis, dc, cfg),
                                                                  "stitch1": curve(st, gt, vis, dc, cfg)})
        times.append(time.time() - t)
    return cur, times


def loader(d, suffix):
    return lru_cache(maxsize=64)(lambda j: I.load(d / f"final_{j}{suffix}.pkl"))


def chosen_items(i, load, stitch_on, pos, groups, cfg):
    return stitch(i, load, pos, groups, cfg.stitch_iou) if stitch_on else load(i)


def write_maps(cfg, shard, its_fn, thr, offset, gate_min, out_dir, png_dir=None, names=None, gt_fn=None):
    rows, fps, counts = [], [], []
    for i in shard:
        canvas, kept, _ = I.paint(its_fn(i), cfg.keep_frac, min(gate_min, thr, thr + offset))
        np.savez_compressed(out_dir / f"{i}.npz", map=canvas.astype(np.uint16), scores=kept)
        ok = np.concatenate([[False], I.size_keep(canvas, kept, thr, offset, cfg.size_split)])
        final = np.where(ok[canvas], canvas, 0)
        counts.append(int(ok.sum()))
        if png_dir is not None:
            Image.fromarray(final.astype(np.uint16)).save(png_dir / names[i])
        if gt_fn is not None:
            gt, vis, sem = gt_fn(i)
            r, f = instance_stats(final, gt, vis, None, sem)
            rows += [(i, *row) for row in r]
            fps += f
    return rows, fps, counts


def test_items(cfg, models, cache, shard, name, d, dev, log):
    times = []
    for k, i in enumerate(shard):
        t = time.time()
        I.save(candidate(sources(models, image_tensor(cache["img_test"][i], dev), cfg, [name]), name, cfg), d / f"final_{i}.pkl")
        times.append(time.time() - t)
        if k % 50 == 0:
            log(f"plant test {k}/{len(shard)} {np.sum(times):.0f} s")
    return times


def global_val_items(cfg, model, cache, idx, dev):
    model.eval()
    with torch.no_grad():
        return {i: global_items(model, image_tensor(cache["img_val"][i], dev), cfg) for i in idx}


@torch.no_grad()
def quick_eval(cfg, model, glob, cache, h, size, dev):
    model.eval()
    cur = []
    for i, g in glob.items():
        gt, vis = val_gt(cache, i)
        z = zoom_sources({"m": model}, image_tensor(cache["img_val"][i], dev), cfg, h, ["m"], size)["m"]
        cur.append(curve(I.merge_zoom(g, z, cfg.z2_gate, contain=True), gt, vis, None, cfg)[cfg.small_offsets.index(0.0)])
    return float(np.nanmax(np.nanmean(np.array(cur), 0)) * 100)
