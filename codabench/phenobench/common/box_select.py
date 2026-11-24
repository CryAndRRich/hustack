import pickle

import numpy as np

from common import box_fusion as F
from common import box_metric as M


class Track:
    def __init__(self, cfg, mod, dirs, cache, names):
        self.cfg, self.mod, self.dirs, self.cache, self.names = cfg, mod, dirs, cache, names

    def mask(self, split, i):
        lab, b, s, ex = self.mod.mask_source(self.dirs, split, i)
        return (lab, F.to_xyxy(b), s), ex

    def yolo(self, split, i, tta):
        z = np.load(self.dirs[f"yolo_{split}"] / f"{self.mod.TASK}_{i}.npz")
        views = [(z[f"{v}|c"] + 1, z[f"{v}|b"], z[f"{v}|s"]) for v in self.cfg.yolo_tta[tta]]
        return views[0] if len(views) == 1 else F.wbf(views, np.ones(len(views)), self.cfg.tta_iou)[:3]

    def fused(self, split, i, p):
        m, ex = self.mask(split, i)
        return F.wbf([m, self.yolo(split, i, p["tta"])], [p["w"], 1.0], p["iou"]), ex

    def features(self, split, i, p):
        f, ex = self.fused(split, i, p)
        lab = np.load(self.dirs[f"sem_{split}"] / f"{i}.npz")["lab"]
        return f, F.features(f, ex, F.integral(lab))

    def date(self, i):
        return self.names[i][:5]


def dets(f):
    return f[0], F.to_cxcywh(f[1]), f[2]


def evaluate(arms, idx, gts, classes):
    rows = [(i, {c: int((gts[i]["labels"] == c).sum()) for c in classes},
             {k: M.image_records(gts[i], fn(i), classes) for k, fn in arms.items()}) for i in idx]
    flat = M.pack(rows)
    return flat, {k: M.summary(flat, k, classes) for k in arms}


def _log_table(table, tag, log):
    for k, r in table.items():
        log(f"{tag} {k}: AP {r['ap']:.2f} (AP_50 {r['ap_50']:.2f}, AP_75 {r['ap_75']:.2f}; per class "
            f"{ {c: None if v is None else round(v, 2) for c, v in r['per_class'].items()} })")


def decide(tr, idx, model_path, log):
    cfg, cls, tag = tr.cfg, tr.mod.CLASSES, f"{tr.mod.TASK} boxes"
    gts = {i: tr.mod.ground_truth(tr.cache, i, M.ground_truth) for i in idx}
    arms = {"mask": lambda i: dets(tr.mask("val", i)[0])}
    arms.update({f"yolo_{t}": (lambda i, t=t: dets(tr.yolo("val", i, t))) for t in cfg.yolo_tta})
    flats, (flat, table) = {}, evaluate(arms, idx, gts, cls)
    flats.update(dict.fromkeys(arms, flat))
    tta = max(cfg.yolo_tta, key=lambda t: table[f"yolo_{t}"]["ap"])
    grid = {f"wbf_t{u}_w{w}": {"tta": tta, "iou": u, "w": w} for u in cfg.wbf_iou for w in cfg.wbf_mask_weights}
    flat, t2 = evaluate({k: (lambda i, p=p: dets(tr.fused("val", i, p)[0])) for k, p in grid.items()}, idx, gts, cls)
    flats.update(dict.fromkeys(t2, flat))
    table.update(t2)
    fkey = max(grid, key=lambda k: table[k]["ap"])
    p = grid[fkey]

    feats = {i: tr.features("val", i, p) for i in idx}
    xs, ys, ds = [], [], []
    for i in idx:
        f, X = feats[i]
        y, keep = F.targets(F.to_cxcywh(f[1]), f[0], gts[i])
        xs.append(X[keep])
        ys.append(y[keep])
        ds += [tr.date(i)] * int(keep.sum())
    X, y, ds = np.concatenate(xs), np.concatenate(ys), np.array(ds)
    dates = sorted(set(tr.date(i) for i in idx))
    folds = {d: F.fit(X[ds != d], y[ds != d]) for d in dates} if len(dates) > 1 else {}
    models = {"folds": folds, "full": F.fit(X, y), "params": p}
    model_path.write_bytes(pickle.dumps(models))
    flat, t3 = evaluate({"rescore": lambda i: rescore_dets(feats[i], models, tr.date(i))}, idx, gts, cls)
    flats["rescore"] = flat
    table.update(t3)
    _log_table(table, tag, log)
    chosen = max(table, key=lambda k: table[k]["ap"])
    fl = flats[chosen]
    by_date = {d: M.summary(fl, chosen, cls, [i for i in idx if tr.date(i) == d]) for d in dates}
    first = M.summary(fl, chosen, cls, [i for i in idx if i < cfg.final_check_images])
    log(f"{tag}: YOLO TTA {tta}, fusion {fkey}, rescoring rows {len(y)}; chosen {chosen} {table[chosen]}")
    return {"key": chosen, **table[chosen], "params": p, "tta": tta, "table": table, "by_date": by_date,
            "first_n": first}


def rescore_dets(fx, models, date):
    f, X = fx
    model = models["folds"].get(date, models["full"]) if date else models["full"]
    return f[0], F.to_cxcywh(f[1]), F.rescored(model, X) if len(X) else f[2]


def final(tr, dec, models, split, i):
    k = dec["key"]
    if k == "mask":
        return dets(tr.mask(split, i)[0])
    if k.startswith("yolo_"):
        return dets(tr.yolo(split, i, k[5:]))
    if k == "rescore":
        return rescore_dets(tr.features(split, i, dec["params"]), models, tr.date(i) if split == "val" else None)
    p = {"tta": dec["tta"], "iou": float(k.split("_t")[1].split("_w")[0]), "w": float(k.split("_w")[1])}
    return dets(tr.fused(split, i, p)[0])
