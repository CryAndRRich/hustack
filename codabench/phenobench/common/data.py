import math
import random
import multiprocessing
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter
from scipy import ndimage

SIZE = 1024
COUNTS = {"train": 1407, "val": 772, "test": 693}
ARRAYS = ([(f"img_{s}", s, "images", np.uint8, (SIZE, SIZE, 3)) for s in COUNTS]
          + [(f"{k}_{s}", s, f, np.uint16, (SIZE, SIZE)) for s in ("train", "val")
             for k, f in (("plant", "plant_instances"), ("leaf", "leaf_instances"))]
          + [("sem_train", "train", "semantics", np.uint8, (SIZE, SIZE)),
             ("pvis_val", "val", "plant_visibility", np.uint8, (SIZE, SIZE)),
             ("lvis_val", "val", "leaf_visibility", np.uint8, (SIZE, SIZE)),
             ("sem_val", "val", "semantics", np.uint8, (SIZE, SIZE))])
I, PID, Y0, X0, Y1, X1, AREA, LEAF, BORDER = range(9)


def find_root(cfg):
    if cfg.data_root:
        return Path(cfg.data_root)
    for base in [Path("/kaggle/input"), Path(__file__).resolve().parents[1] / "data"]:
        if base.exists():
            hits = sorted(p.parent.parent for p in base.glob("**/PhenoBench/train/images"))
            if hits:
                return hits[0]
    raise FileNotFoundError("PhenoBench/train/images not found under /kaggle/input or data/")


def list_images(root, split):
    return sorted(p.name for p in (root / split / "images").glob("*.png"))


def _fill(task):
    path, folder, dtype, shape, items = task
    mm = np.load(path, mmap_mode="r+")
    for i, f in items:
        im = Image.open(folder / f)
        a = np.asarray(im.convert("RGB") if len(shape) == 3 else im)
        assert a.shape == shape, f"{folder / f}: {a.shape}"
        if a.dtype != dtype:
            assert a.max() <= np.iinfo(dtype).max, f"{folder / f}: {a.dtype} values exceed {dtype}"
            a = a.astype(dtype)
        mm[i] = a
    mm.flush()
    del mm
    return len(items)


def _boxes(task):
    plant_path, leaf_path, idx = task
    plant, leaf = np.load(plant_path, mmap_mode="r"), np.load(leaf_path, mmap_mode="r")
    out = []
    for i in idx:
        p, lf = np.asarray(plant[i]).astype(np.int32), np.asarray(leaf[i])
        objs = ndimage.find_objects(p)
        areas = np.bincount(p.ravel())
        leafy = set(np.unique(p[lf > 0]).tolist())
        for pid, sl in enumerate(objs, 1):
            if sl is None:
                continue
            y0, x0, y1, x1 = sl[0].start, sl[1].start, sl[0].stop, sl[1].stop
            border = int(y0 == 0 or x0 == 0 or y1 == SIZE or x1 == SIZE)
            out.append((i, pid, y0, x0, y1, x1, int(areas[pid]), int(pid in leafy), border))
    return out


def build_cache(root, cache, names, log, procs=4):
    cache.mkdir(parents=True, exist_ok=True)
    done = cache / "done"
    if done.exists():
        return
    tasks = []
    for key, split, folder, dtype, shape in ARRAYS:
        path = cache / f"{key}.npy"
        np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=(len(names[split]),) + shape).flush()
        items = list(enumerate(names[split]))
        tasks += [(path, root / split / folder, dtype, shape, items[j:j + 32]) for j in range(0, len(items), 32)]
    n = 0
    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(procs) as pool:
        for k in pool.imap_unordered(_fill, tasks):
            n += k
    log(f"cache: {n} PNGs decoded into {len(ARRAYS)} memmaps")
    idx = list(range(len(names["train"])))
    with ctx.Pool(procs) as pool:
        boxes = [b for part in pool.map(_boxes, [(cache / "plant_train.npy", cache / "leaf_train.npy", idx[j::procs])
                                                for j in range(procs)]) for b in part]
    boxes = np.array(sorted(boxes), dtype=np.int64)
    np.save(cache / "plant_boxes.npy", boxes)
    log(f"cache: {len(boxes)} train plants, {int(boxes[:, LEAF].sum())} with leaves, "
        f"{int((boxes[:, BORDER] == 0).sum())} away from the border, {int((boxes[:, AREA] < 2500).sum())} < 2500 px")
    done.write_text("ok")


def open_cache(cache):
    return {key: np.load(cache / f"{key}.npy", mmap_mode="r") for key, *_ in ARRAYS}


def colour_jitter(img):
    mean = img.mean()
    img = (img - mean) * random.uniform(0.7, 1.3) + mean * random.uniform(0.7, 1.3)
    gray = img.mean(axis=2, keepdims=True)
    img = gray + (img - gray) * random.uniform(0.7, 1.3)
    img = img * np.random.uniform(0.9, 1.1, size=(1, 1, 3)).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


def resize(a, size, nearest):
    if nearest:
        return np.asarray(Image.fromarray(np.ascontiguousarray(a).astype(np.int32)).resize((size, size), Image.NEAREST)).astype(np.int32)
    return np.asarray(Image.fromarray(np.ascontiguousarray(a)).resize((size, size), Image.BILINEAR))


def blur(img, p):
    if random.random() < p:
        return np.asarray(Image.fromarray(img).filter(ImageFilter.GaussianBlur(random.uniform(0.5, 1.5))))
    return img


def rot_flip(img, ids):
    k = random.randrange(4)
    img, ids = np.rot90(img, k), np.rot90(ids, k)
    if random.random() < 0.5:
        img, ids = img[:, ::-1], ids[:, ::-1]
    return img, ids


def to_tensors(img, ids, stride):
    return (torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))),
            torch.from_numpy(np.ascontiguousarray(ids[::stride, ::stride]).astype(np.int32)))


def size_weights(small, w):
    p = np.where(small, w, 1.0)
    return p / p.sum()


class ZoomSet(torch.utils.data.Dataset):
    def __init__(self, cfg, cache_dir, out_size, stride):
        self.cfg, self.cache_dir, self.out_size, self.stride, self.mm = cfg, cache_dir, out_size, stride, None
        self.boxes = np.load(cache_dir / "plant_boxes.npy")
        self.box_p = size_weights(self.boxes[:, AREA] < cfg.small_area, cfg.small_weight)
        self.paste = self.boxes[self.boxes[:, BORDER] == 0]
        self.paste_p = size_weights(self.paste[:, AREA] < cfg.small_area, cfg.small_weight)
        self.max_instances = int(np.bincount(self.boxes[:, I]).max()) + cfg.paste_max

    def __len__(self):
        return len(self.boxes)

    def load(self, n):
        if self.mm is None:
            self.mm = {k: np.load(self.cache_dir / f"{k}_train.npy", mmap_mode="r") for k in ("img", "plant")}
        return np.array(self.mm["img"][n]), np.asarray(self.mm["plant"][n]).astype(np.int32)

    def paste_plants(self, img, ids):
        nxt = int(ids.max()) + 1
        for row in self.paste[np.random.choice(len(self.paste), random.randint(1, self.cfg.paste_max), p=self.paste_p)]:
            src = np.asarray(self.mm["img"][row[I], row[Y0]:row[Y1], row[X0]:row[X1]])
            m = np.asarray(self.mm["plant"][row[I], row[Y0]:row[Y1], row[X0]:row[X1]]) == row[PID]
            k, flip = random.randrange(4), random.random() < 0.5
            src, m = np.rot90(src, k), np.rot90(m, k)
            if flip:
                src, m = src[:, ::-1], m[:, ::-1]
            h, w = m.shape
            for _ in range(5):
                y, x = random.randint(0, SIZE - h), random.randint(0, SIZE - w)
                if not ids[y:y + h, x:x + w][m].any():
                    img[y:y + h, x:x + w][m] = src[m]
                    ids[y:y + h, x:x + w][m] = nxt
                    nxt += 1
                    break
        return img, ids

    def __getitem__(self, _):
        cfg = self.cfg
        row = self.boxes[np.random.choice(len(self.boxes), p=self.box_p)]
        img, ids = self.load(row[I])
        if random.random() < cfg.paste_p:
            img, ids = self.paste_plants(img, ids)
        z = math.exp(random.uniform(math.log(cfg.zoom_min), math.log(cfg.zoom_max)))
        w = min(SIZE, round(self.out_size / z))
        y, x = window_start(row[Y0], row[Y1], w), window_start(row[X0], row[X1], w)
        img, ids = rot_flip(img[y:y + w, x:x + w], ids[y:y + w, x:x + w])
        img, ids = resize(img, self.out_size, False), resize(ids, self.out_size, True)
        img = blur(np.ascontiguousarray(img), cfg.blur_p)
        return to_tensors(colour_jitter(img.astype(np.float32)), ids, self.stride)


def window_start(a, b, w):
    lo, hi = (b - w, a) if b - a <= w else (a, b - w)
    return min(max(random.randint(lo, hi), 0), SIZE - w)


def disk(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r


def morph(mask, r):
    if r > 0:
        return ndimage.binary_dilation(mask, structure=disk(r))
    if r < 0:
        return ndimage.binary_erosion(mask, structure=disk(-r))
    return mask


def crop_window(bbox, cfg):
    y0, x0, y1, x1 = (int(v) for v in bbox)
    s = min(max(cfg.leaf_fill * cfg.leaf_size / max(y1 - y0, x1 - x0, 1), cfg.leaf_zoom_min), cfg.leaf_zoom_max)
    side = int(math.ceil(cfg.leaf_size / s))
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    return cy - side // 2, cx - side // 2, side


def extract(arr, top, left, side):
    out = np.zeros((side, side) + arr.shape[2:], arr.dtype)
    ys, ye, xs, xe = max(top, 0), min(top + side, arr.shape[0]), max(left, 0), min(left + side, arr.shape[1])
    if ys < ye and xs < xe:
        out[ys - top:ye - top, xs - left:xe - left] = arr[ys:ye, xs:xe]
    return out


def leaf_crop(img, keep, cfg):
    return resize(img * keep[..., None].astype(np.uint8), cfg.leaf_size, False)
