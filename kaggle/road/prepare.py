import json
import os
import shutil
import time
from multiprocessing.pool import ThreadPool
from pathlib import Path

import pandas as pd
from PIL import Image

CLASS_NAMES = ["D00", "D10", "D20", "D40"]
SPLIT_SIZES = {"train": 14691, "val": 2150}
SMOKE_IMAGES = {"train": 64, "val": 32}


def find_kit():
    root = Path("/kaggle/input")
    for pattern in ["*/train_labels.csv", "*/*/train_labels.csv", "*/*/*/train_labels.csv"]:
        for p in sorted(root.glob(pattern)):
            kit = p.parent
            if (kit / "train_images").is_dir() and (kit / "score.py").exists() and (kit / "test_image_ids.csv").exists():
                return kit
    raise FileNotFoundError("competition data not found under /kaggle/input (attach the competition)")


def copy_split(src, dst, ids):
    dst.mkdir(parents=True, exist_ok=True)

    def one(iid):
        s, d = src / f"{iid}.jpg", dst / f"{iid}.jpg"
        if not d.exists() or d.stat().st_size != s.stat().st_size:
            shutil.copyfile(s, d)
        with Image.open(d) as im:
            return im.size

    with ThreadPool(32) as pool:
        return dict(zip(ids, pool.map(one, ids)))


def clipped_boxes(labels, sizes):
    groups = {k: g for k, g in labels.groupby("image_id")}
    unknown = set(groups) - set(sizes)
    assert not unknown, f"{len(unknown)} labelled ids without an image, e.g. {sorted(unknown)[:3]}"
    out = {}
    for iid, (w, h) in sizes.items():
        rows = []
        g = groups.get(iid)
        if g is not None:
            for c, x1, y1, x2, y2 in g[["class_id", "x_min", "y_min", "x_max", "y_max"]].itertuples(index=False):
                x1, x2 = max(0.0, x1), min(float(w), x2)
                y1, y2 = max(0.0, y1), min(float(h), y2)
                if x2 > x1 and y2 > y1:
                    rows.append((int(c), x1, y1, x2, y2))
        out[iid] = rows
    return out


def write_yolo(boxes, sizes, dst):
    dst.mkdir(parents=True, exist_ok=True)
    for iid, rows in boxes.items():
        w, h = sizes[iid]
        lines = [f"{c} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"
                 for c, x1, y1, x2, y2 in rows]
        (dst / f"{iid}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))


def write_coco(boxes, sizes, ids, path):
    images, annotations = [], []
    for k, iid in enumerate(ids):
        w, h = sizes[iid]
        images.append({"id": k, "file_name": f"{iid}.jpg", "width": w, "height": h})
        for c, x1, y1, x2, y2 in boxes[iid]:
            annotations.append({"id": len(annotations), "image_id": k, "category_id": c, "iscrowd": 0,
                                "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1)})
    categories = [{"id": i, "name": n} for i, n in enumerate(CLASS_NAMES)]
    path.write_text(json.dumps({"images": images, "annotations": annotations, "categories": categories}))
    return len(annotations)


def prepare(cfg, kit):
    t0 = time.time()
    data = Path(cfg.data_dir)
    splits = {}
    for split in SPLIT_SIZES:
        src = kit / f"{split}_images"
        ids = sorted(f[:-4] for f in os.listdir(src) if f.endswith(".jpg"))
        assert len(ids) == SPLIT_SIZES[split], f"{split}: {len(ids)} images, expected {SPLIT_SIZES[split]}"
        sizes = copy_split(src, data / "images" / split, ids)
        labels = pd.read_csv(kit / f"{split}_labels.csv")
        boxes = clipped_boxes(labels, sizes)
        n_boxes = sum(len(r) for r in boxes.values())
        assert n_boxes == len(labels), f"{split}: kept {n_boxes} of {len(labels)} boxes"
        write_yolo(boxes, sizes, data / "labels" / split)
        n_coco = write_coco(boxes, sizes, ids[:SMOKE_IMAGES[split]] if cfg.smoke else ids, data / f"{split}.json")
        splits[split] = {"images": len(ids), "boxes": n_boxes, "with_boxes": int(labels.image_id.nunique()),
                         "coco_boxes": n_coco}
    names = "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASS_NAMES))
    yaml_path = data / "data.yaml"
    yaml_path.write_text(f"path: {data}\ntrain: images/train\nval: images/val\nnames:\n{names}")
    total, used, free = shutil.disk_usage(data)
    print(f"prepare: {splits} in {time.time() - t0:.0f}s; disk free {free / 2**30:.1f} of {total / 2**30:.1f} GB")
    return data
