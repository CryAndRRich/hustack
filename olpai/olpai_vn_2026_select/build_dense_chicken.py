import glob
import json
import os
import zipfile
from pathlib import Path

import numpy as np
from scipy.special import erf

ROOT = Path(__file__).resolve().parent.parent / "data"
SRC = ROOT / "external" / "dense_chicken"
OUT = ROOT / "dense-chicken.zip"
H, W, CELL, SIGMA = 720, 1280, 5, 24 / 3.2


def axis_mass(c, n):
    edges = np.arange(n + 1) * CELL
    return np.diff(0.5 * (1 + erf((edges[None] - c[:, None]) / (np.sqrt(2) * SIGMA))), axis=1)


def render(pts):
    return (axis_mass(pts[:, 1], H // CELL).T @ axis_mass(pts[:, 0], W // CELL)).astype(np.float32)


def main():
    images = {Path(f).stem: f for f in glob.glob(str(SRC / "images" / "*" / "*.jpg"))}
    ids = sorted(images, key=int)
    maps, counts = [], []
    for i in ids:
        d = json.load(open(SRC / "json" / f"{i}.json"))
        assert (d["imageWidth"], d["imageHeight"]) == (W, H)
        pts = np.array([np.mean(s["points"], 0) for s in d["shapes"] if s["label"] == "chicken"])
        assert len(pts) == len(d["shapes"]) and pts.min() >= 0 and (pts < [W, H]).all()
        maps.append(render(pts))
        counts.append(len(pts))
    maps, counts = np.stack(maps), np.array(counts)
    loss = 1 - maps.sum((1, 2)) / counts
    print(f"{len(ids)} images | {counts.sum()} chickens, per image {counts.min()}-{counts.max()} "
          f"(median {np.median(counts):.0f}) | border mass loss mean {loss.mean():.4f} max {loss.max():.4f}")
    np.savez_compressed(ROOT / "dc_maps.npz", ids=np.array(ids), maps=maps, counts=counts)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_STORED) as z:
        z.write(ROOT / "dc_maps.npz", "dc_maps.npz")
        for i in ids:
            z.write(images[i], f"dc_{i}.jpg")
    os.remove(ROOT / "dc_maps.npz")
    print(f"wrote {OUT} ({OUT.stat().st_size / 2**20:.0f} MB)")


if __name__ == "__main__":
    main()
