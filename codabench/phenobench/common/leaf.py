import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage

from common import items as I
from common.data import crop_window, extract, leaf_crop, morph
from common.fuse import fuse, unview, view
from common.models import predict_batch
from common.plant import Curves, curve
from common.pq import instance_stats, leaf_dont_care

SIZE = 1024


@torch.no_grad()
def crop_items(cfg, model, img, plant_map, scores, dev, tile):
    objs = ndimage.find_objects(plant_map.astype(np.int32))
    crops, metas = [], []
    for pid, sl in enumerate(objs, 1):
        if sl is None:
            continue
        top, left, side = crop_window((sl[0].start, sl[1].start, sl[0].stop, sl[1].stop), cfg)
        m = extract(plant_map, top, left, side) == pid
        if m.sum() < cfg.min_plant_area:
            continue
        keep = morph(m, cfg.leaf_dilate)
        crops.append(leaf_crop(extract(img, top, left, side), keep, cfg))
        metas.append((top, left, side, keep, (float(scores[pid - 1]), (int(tile), pid))))
    out = []
    n_views = cfg.leaf_views
    group = max(1, cfg.leaf_infer_batch // n_views)
    for a in range(0, len(crops), group):
        idx = list(range(a, min(a + group, len(crops))))
        x = torch.from_numpy(np.stack([crops[i] for i in idx])).to(dev).permute(0, 3, 1, 2).float()
        outs = predict_batch(model, torch.stack([view(x[k], v) for k in range(len(idx)) for v in range(n_views)]),
                             cfg, cfg.leaf_size)
        for k, i in enumerate(idx):
            views = [(unview(p, v), s) for v, (p, s) in enumerate(outs[k * n_views:(k + 1) * n_views])]
            prob, score = fuse(views, n_views, cfg.fuse_iou)
            out += paste(cfg, prob, score, metas[i], dev)
    return I.by_score(out)


def paste(cfg, prob, score, meta, dev):
    top, left, side, keep, tag = meta
    sel = score >= cfg.min_score
    if not bool(sel.any()):
        return []
    prob, sc = prob[sel], score[sel].float().cpu().numpy()
    big = ((F.interpolate(prob[:, None].float(), size=(side, side), mode="bilinear", align_corners=False)[:, 0] > 0.5)
           & torch.from_numpy(keep).to(dev)).cpu().numpy()
    rows, cols = big.any(2), big.any(1)
    out = []
    for k in range(len(sc)):
        r, c = np.flatnonzero(rows[k]), np.flatnonzero(cols[k])
        if len(r) == 0:
            continue
        m = big[k, r[0]:r[-1] + 1, c[0]:c[-1] + 1]
        pl = I.place(I.largest_component(m), top + int(r[0]), left + int(c[0]))
        if pl is not None:
            out.append((pl[0], pl[1], pl[2], float(sc[k]), tag))
    return out


def gated(its, gate):
    return [it for it in its if it[4][0] >= gate]


def val_gt(cache, i):
    gt, vis = np.asarray(cache["leaf_val"][i]), np.asarray(cache["lvis_val"][i])
    return gt, vis, leaf_dont_care(np.asarray(cache["plant_val"][i]), gt)


def plant_map(d, i):
    z = np.load(d / f"{i}.npz")
    return z["map"].astype(np.int32), z["scores"]


def val_pass(cfg, model, cache, shard, maps_dir, d, gates, dev, log):
    cur, times = Curves(len(shard)), []
    for k, i in enumerate(shard):
        t = time.time()
        pm, ps = plant_map(maps_dir, i)
        its = crop_items(cfg, model, np.asarray(cache["img_val"][i]), pm, ps, dev, i)
        I.save(its, d / f"{i}.pkl")
        gt, vis, dc = val_gt(cache, i)
        cur.add(k, lambda its=its, gt=gt, vis=vis, dc=dc: {f"g{g}": curve(gated(its, g), gt, vis, dc, cfg) for g in gates})
        times.append(time.time() - t)
        if k % 50 == 0:
            log(f"leaf val {k}/{len(shard)} {np.sum(times):.0f} s")
    return cur, times


def write(cfg, shard, its_fn, gate, thr, offset, png_dir=None, names=None, gt_fn=None):
    rows, fps, counts = [], [], []
    for i in shard:
        canvas, kept, _ = I.paint(gated(its_fn(i), gate), cfg.keep_frac, min(thr, thr + offset))
        ok = np.concatenate([[False], I.size_keep(canvas, kept, thr, offset, cfg.size_split)])
        canvas = np.where(ok[canvas], canvas, 0)
        counts.append(int(ok.sum()))
        if png_dir is not None:
            Image.fromarray(canvas.astype(np.uint16)).save(png_dir / names[i])
        if gt_fn is not None:
            gt, vis, dc = gt_fn(i)
            r, f = instance_stats(canvas, gt, vis, dc)
            rows += [(i, *row) for row in r]
            fps += f
    return rows, fps, counts


def test_items(cfg, model, cache, shard, maps_dir, d, dev, log):
    times = []
    for k, i in enumerate(shard):
        t = time.time()
        pm, ps = plant_map(maps_dir, i)
        I.save(crop_items(cfg, model, np.asarray(cache["img_test"][i]), pm, ps, dev, i), d / f"{i}.pkl")
        times.append(time.time() - t)
        if k % 50 == 0:
            log(f"leaf test {k}/{len(shard)} {np.sum(times):.0f} s")
    return times
