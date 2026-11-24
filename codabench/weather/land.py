import struct

import numpy as np
import scipy.ndimage as ndi
import shapely
import torch
from PIL import Image, ImageDraw

from config import LAND_RES_DEG


def _inside(pt, ring):
    x, y = pt
    x1, y1, x2, y2 = ring[:-1, 0], ring[:-1, 1], ring[1:, 0], ring[1:, 1]
    cross = (y1 > y) != (y2 > y)
    xs = x1 + (y - y1) * (x2 - x1) / np.where(cross, y2 - y1, 1.0)
    return bool(np.sum(cross & (x < xs)) % 2)


def read_polygons(shp_path):
    data = shp_path.read_bytes()
    pos, shells, holes = 100, [], []
    while pos + 8 <= len(data):
        _, words = struct.unpack(">ii", data[pos:pos + 8])
        rec = data[pos + 8:pos + 8 + 2 * words]
        pos += 8 + 2 * words
        if struct.unpack("<i", rec[:4])[0] != 5:
            continue
        nparts, npts = struct.unpack("<ii", rec[36:44])
        parts = list(np.frombuffer(rec, "<i4", nparts, 44)) + [npts]
        pts = np.frombuffer(rec, "<f8", 2 * npts, 44 + 4 * nparts).reshape(-1, 2)
        for s, e in zip(parts[:-1], parts[1:]):
            ring = pts[s:e]
            area = np.sum(ring[:-1, 0] * ring[1:, 1] - ring[1:, 0] * ring[:-1, 1])
            (shells if area < 0 else holes).append(ring)
    polys = [(sh, []) for sh in shells]
    boxes = np.array([[sh[:, 0].min(), sh[:, 1].min(), sh[:, 0].max(), sh[:, 1].max()] for sh in shells])
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    for hole in holes:
        x, y = hole[0]
        cand = np.where((boxes[:, 0] <= x) & (x <= boxes[:, 2]) & (boxes[:, 1] <= y) & (y <= boxes[:, 3]))[0]
        owners = [k for k in cand[np.argsort(areas[cand])] if _inside((x, y), shells[k])]
        if owners:
            polys[owners[0]][1].append(hole)
        else:
            polys.append((hole[::-1], []))
    return polys


def land_union(polys):
    geoms = [shapely.Polygon(s, h) for s, h in polys]
    bad = [i for i, g in enumerate(geoms) if not g.is_valid]
    for i in bad:
        geoms[i] = shapely.make_valid(geoms[i])
    print(f"land: {len(geoms)} polygons, {sum(len(h) for _, h in polys)} holes, {len(bad)} repaired", flush=True)
    geom = shapely.unary_union(geoms)
    shapely.prepare(geom)
    return geom


def on_land(union, lat, lon):
    return shapely.contains_xy(union, np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64))


class LandRaster:
    def __init__(self, polys, corr, device):
        self.lat0, self.lat1 = float(corr.lat[0]), float(corr.lat[-1])
        self.lon0, self.lon1 = float(corr.lon[0]), float(corr.lon[-1])
        res = LAND_RES_DEG
        self.h = int(round((self.lat1 - self.lat0) / res)) + 1
        self.w = int(round((self.lon1 - self.lon0) / res)) + 1
        img = Image.new("L", (self.w, self.h), 0)
        draw = ImageDraw.Draw(img)

        def px(ring):
            lon = ring[:, 0] + 360.0 if corr.lon_shift and ring[:, 0].max() < 0 else ring[:, 0]
            return [((x - self.lon0) / res, (y - self.lat0) / res) for x, y in zip(lon, ring[:, 1])]

        for shell, holes in polys:
            lon = shell[:, 0] + 360.0 if corr.lon_shift and shell[:, 0].max() < 0 else shell[:, 0]
            if lon.max() < self.lon0 - 1 or lon.min() > self.lon1 + 1:
                continue
            if shell[:, 1].max() < self.lat0 - 1 or shell[:, 1].min() > self.lat1 + 1:
                continue
            draw.polygon(px(shell), fill=1)
            for hole in holes:
                draw.polygon(px(hole), fill=0)
        mask = np.array(img, dtype=bool)
        coslat = np.cos(np.radians(max(abs(self.lat0), abs(self.lat1))))
        dist = ndi.distance_transform_edt(~mask, sampling=(res * 111.195, res * 111.195 * coslat))
        self.dist = torch.tensor(dist, dtype=torch.float32, device=device)
        print(f"land raster {corr.name}: {self.h}x{self.w}, land cells {mask.mean():.3f}", flush=True)

    def km(self, lat, lon_normed):
        fy = ((lat - self.lat0) / (self.lat1 - self.lat0) * (self.h - 1)).clamp(0, self.h - 1)
        fx = ((lon_normed - self.lon0) / (self.lon1 - self.lon0) * (self.w - 1)).clamp(0, self.w - 1)
        y0 = torch.floor(fy).clamp(max=self.h - 2)
        x0 = torch.floor(fx).clamp(max=self.w - 2)
        wy, wx = (fy - y0).to(self.dist.dtype), (fx - x0).to(self.dist.dtype)
        y0, x0 = y0.long(), x0.long()
        d = self.dist
        top = (1 - wx) * d[y0, x0] + wx * d[y0, x0 + 1]
        bot = (1 - wx) * d[y0 + 1, x0] + wx * d[y0 + 1, x0 + 1]
        return ((1 - wy) * top + wy * bot).to(lat.dtype)
