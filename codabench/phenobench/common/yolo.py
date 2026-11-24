import numpy as np
from PIL import Image

from common.box_metric import SIZE


def export(cache, labels_fn, names, root, n, n_val=20):
    for sub in ("images/train", "labels/train", "images/val", "labels/val"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    for i in range(n):
        cls, b = labels_fn(cache, i)
        keep = cls >= 0
        rows = "".join(f"{c} {x / SIZE:.6f} {y / SIZE:.6f} {max(w, 1) / SIZE:.6f} {max(h, 1) / SIZE:.6f}\n"
                       for c, (x, y, w, h) in zip(cls[keep], b[keep]))
        img = Image.fromarray(np.asarray(cache["img_train"][i]))
        for sub in ["train"] + (["val"] if i < n_val else []):
            img.save(root / "images" / sub / f"{i}.png", compress_level=1)
            (root / "labels" / sub / f"{i}.txt").write_text(rows)
    (root / "data.yaml").write_text(f"path: {root}\ntrain: images/train\nval: images/val\nnames:\n"
                                    + "".join(f"  {k}: {v}\n" for k, v in names.items()))


def train(cfg, root, out, hours):
    from ultralytics import YOLO
    model = YOLO(cfg.yolo_model)
    model.train(data=str(root / "data.yaml"), imgsz=cfg.yolo_imgsz, epochs=1000, time=hours, batch=cfg.yolo_batch,
                device=0, workers=cfg.yolo_workers, project=str(out), name="run", exist_ok=True, val=False, plots=False,
                cache="ram", flipud=0.5, seed=0, verbose=False)
    return out / "run" / "weights" / "last.pt"


def predict(model, img, views, conf, max_det):
    out = {}
    for v in views:
        size, flip = int(v.rstrip("f")), v.endswith("f")
        x = np.ascontiguousarray((img[:, ::-1] if flip else img)[..., ::-1])
        r = model.predict(x, imgsz=size, conf=conf, max_det=max_det, verbose=False, half=True)[0].boxes
        b = r.xyxy.cpu().numpy().astype(np.float64)
        if flip:
            b[:, [0, 2]] = SIZE - b[:, [2, 0]]
        out[v] = (r.cls.cpu().numpy().astype(np.int64), b, r.conf.cpu().numpy().astype(np.float64))
    return out
