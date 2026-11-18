import io
import json
import re
import subprocess
import sys
import tarfile
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from preprocess import METALS, OUT, SITE, load, log

GRID_URLS = ["https://github.com/r-spatial/gstat/raw/main/data/jura.rda",
             "https://cran.r-project.org/src/contrib/Archive/gstat/gstat_2.1-1.tar.gz"]
API = "https://api3.geo.admin.ch/rest/services"
STAC = "https://data.geo.admin.ch/api/stac/v0.9"
GEOLOGY_LAYERS = ["ch.swisstopo.geologie-geocover", "ch.swisstopo.geologie-geologischer_atlas"]
EXTRA_LAYER_SEARCHES = ["geochemischer bodenatlas", "bodeneignung"]
AEF_BASE = "https://data.source.coop/tge-labs/aef/"
AEF_INDEX = AEF_BASE + "v1/annual/aef_index_stac_geoparquet.parquet"
AEF_YEARS = range(2017, 2026)
DEM_RES, DEM_MARGIN = 2.0, 600.0
TRANSECT_MIN_TEST_KM = 0.01
OUT_FILE = "cobalt_sites_v3.csv"


def fetch(url, timeout=120):
    req = urllib.request.Request(url, headers={"User-Agent": "kaggle-notebook"})
    for _ in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:
            err = e
    raise RuntimeError(f"{url}: {err}")


def fetch_json(url):
    return json.loads(fetch(url))


def dist_matrix(a, b):
    return np.hypot(a[:, None, 0] - b[None, :, 0], a[:, None, 1] - b[None, :, 1])


def jura_rda():
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyreadr", "rasterio"], check=True)
    import pyreadr
    path = OUT / "jura.rda"
    for url in GRID_URLS:
        try:
            raw = fetch(url)
        except RuntimeError as e:
            log(f"skip {e}")
            continue
        if url.endswith(".tar.gz"):
            with tarfile.open(fileobj=io.BytesIO(raw)) as t:
                raw = t.extractfile("gstat/data/jura.rda").read()
        path.write_bytes(raw)
        break
    assert path.exists(), "could not download jura.rda"
    cols = {o["object_name"]: list(o["columns"]) for o in pyreadr.list_objects(str(path))}
    log(f"jura.rda objects and column names (no values): {cols}")
    return pyreadr, path, cols


def factor_codes(df):
    for c in ("Rock", "Landuse"):
        if c in df and isinstance(df[c].dtype, pd.CategoricalDtype):
            df[c] = df[c].cat.codes.astype(int) + 1
    return df


def jura_grid(pyreadr, path):
    g = factor_codes(pyreadr.read_r(str(path), use_objects=["jura.grid"])["jura.grid"])
    assert {"Xloc", "Yloc", "Landuse", "Rock", "long", "lat"} <= set(g.columns)
    assert not set(g.columns) & set(METALS), "jura.grid must hold no metal values"
    return g[["Xloc", "Yloc", "Landuse", "Rock", "long", "lat"]].astype(float).astype({"Landuse": int, "Rock": int})


def transect(pyreadr, path, cols, train, test, g):
    name = "transect.dat"
    if name not in cols or not {"Xloc", "Yloc", "Co"} <= set(cols[name]):
        log(f"transect.dat: missing or lacks Xloc/Yloc/Co (columns {cols.get(name)}) -> not used")
        return None
    t = factor_codes(pyreadr.read_r(str(path), use_objects=[name])[name]).reset_index(drop=True)
    xy = t[["Xloc", "Yloc"]].to_numpy(float)
    xy_tr = train[["Xloc", "Yloc"]].to_numpy(float)
    inside = ((xy[:, 0] >= g["Xloc"].min()) & (xy[:, 0] <= g["Xloc"].max())
              & (xy[:, 1] >= g["Yloc"].min()) & (xy[:, 1] <= g["Yloc"].max()))
    d_test = dist_matrix(xy, test[["Xloc", "Yloc"]].to_numpy(float)).min(1)
    near_test = d_test < TRANSECT_MIN_TEST_KM
    log(f"transect.dat: {len(t)} rows; outside the grid area {int((~inside).sum())}; within "
        f"{TRANSECT_MIN_TEST_KM * 1000:.0f} m of a test site {int(near_test.sum())} (dropped before any value is used)")
    t, d_test = t[inside & ~near_test].reset_index(drop=True), d_test[inside & ~near_test]
    d_tr = dist_matrix(t[["Xloc", "Yloc"]].to_numpy(float), xy_tr)
    dup = (d_tr.min(1) < 0.001) & np.isclose(t["Co"].to_numpy(float), train["Co"].to_numpy()[d_tr.argmin(1)])
    keep = ~dup & t["Co"].notna().to_numpy()
    log(f"transect.dat: duplicates of train rows {int(dup.sum())}, missing Co {int(t['Co'].isna().sum())}, kept {int(keep.sum())}")
    t, d_test, d_tr = t[keep].reset_index(drop=True), d_test[keep], d_tr[keep]
    if t.empty:
        return None
    log(f"transect kept rows: nearest test site km quantiles 0/50/100 % {np.round(np.quantile(d_test, [0, .5, 1]), 3).tolist()}, "
        f"nearest train site {np.round(np.quantile(d_tr.min(1), [0, .5, 1]), 3).tolist()}")
    gi = dist_matrix(t[["Xloc", "Yloc"]].to_numpy(float), g[["Xloc", "Yloc"]].to_numpy()).argmin(1)
    for c in ("Rock", "Landuse"):
        grid_vals = pd.Series(g[c].to_numpy()[gi], index=t.index)
        if c not in t or t[c].isna().any():
            log(f"transect.dat: {c} filled from the nearest jura.grid cell where missing")
            t[c] = grid_vals if c not in t else t[c].fillna(grid_vals)
    return t[SITE + [m for m in METALS if m in t]].astype({"Rock": int, "Landuse": int})


def to_lv95(g):
    from pyproj import Transformer
    e, n = Transformer.from_crs(4326, 2056, always_xy=True).transform(g["long"].to_numpy(), g["lat"].to_numpy())
    a = np.column_stack([g["Xloc"], g["Yloc"], np.ones(len(g))])
    coef, *_ = np.linalg.lstsq(a, np.column_stack([e, n]), rcond=None)
    rmse = np.sqrt(((a @ coef - np.column_stack([e, n])) ** 2).sum(1).mean())
    log(f"local km -> LV95 affine fit: RMSE {rmse:.2f} m")
    assert rmse < 20, "local grid -> LV95 is not affine"
    return lambda d: np.column_stack([d["Xloc"], d["Yloc"], np.ones(len(d))]) @ coef


def lv95_to_wgs(en):
    from pyproj import Transformer
    return np.column_stack(Transformer.from_crs(2056, 4326, always_xy=True).transform(en[:, 0], en[:, 1]))


def neighbourhood(sites, g):
    d = dist_matrix(sites[["Xloc", "Yloc"]].to_numpy(float), g[["Xloc", "Yloc"]].to_numpy())
    out = {}
    for radius in (0.1, 0.25, 0.5):
        w = d <= radius
        for r in range(1, 6):
            out[f"rock{r}_share_{int(radius * 1000)}m"] = (w & (g["Rock"].to_numpy() == r)[None]).sum(1) / w.sum(1)
    for r in range(1, 6):
        out[f"dist_rock{r}"] = np.where(g["Rock"].to_numpy()[None] == r, d, np.inf).min(1)
    return pd.DataFrame(out)


def identify(layer, en):
    e, n = en
    url = (f"{API}/api/MapServer/identify?geometry={e:.1f},{n:.1f}&geometryType=esriGeometryPoint&layers=all:{layer}"
           f"&mapExtent={e - 100:.0f},{n - 100:.0f},{e + 100:.0f},{n + 100:.0f}&imageDisplay=200,200,96&tolerance=0"
           f"&returnGeometry=false&sr=2056&lang=en")
    out = {}
    for hit in fetch_json(url)["results"][:1]:
        for k, v in hit.get("attributes", hit.get("properties", {})).items():
            out[f"{layer.split('.')[-1]}:{k}"] = v
    return out


def layer_table(layers, en):
    with ThreadPoolExecutor(16) as ex:
        rows = list(ex.map(lambda p: {k: v for layer in layers for k, v in identify(layer, p).items()},
                           [tuple(p) for p in en]))
    return pd.DataFrame(rows, index=range(len(en)))


def stac_items(collection, bbox):
    url = f"{STAC}/collections/{collection}/items?bbox={','.join(f'{v:.5f}' for v in bbox)}&limit=100"
    items = []
    while url:
        page = fetch_json(url)
        items += page["features"]
        url = next((link["href"] for link in page.get("links", []) if link.get("rel") == "next"), None)
    return items


def wgs_bbox(en, margin=0.0):
    lo, hi = lv95_to_wgs(en.min(0, keepdims=True) - margin)[0], lv95_to_wgs(en.max(0, keepdims=True) + margin)[0]
    return np.r_[lo, hi]


def dem_terrain(en):
    import rasterio
    from scipy import ndimage
    e0, n0 = en.min(0) - DEM_MARGIN
    e1, n1 = en.max(0) + DEM_MARGIN
    latest = {}
    for it in stac_items("ch.swisstopo.swissalti3d", wgs_bbox(en, DEM_MARGIN + 1000)):
        for a in it["assets"].values():
            if a["href"].endswith("_2_2056_5728.tif"):
                _, year, tile = it["id"].split("_")[:3]
                if tile not in latest or year > latest[tile][0]:
                    latest[tile] = (year, a["href"])
    log(f"swissALTI3D 2 m: {len(latest)} tiles, years {sorted({y for y, _ in latest.values()})}")
    assert latest, "no swissALTI3D tiles found"
    w, h = int(np.ceil((e1 - e0) / DEM_RES)), int(np.ceil((n1 - n0) / DEM_RES))
    z = np.full((h, w), np.nan, np.float32)

    def place(href):
        with rasterio.MemoryFile(fetch(href)) as mf, mf.open() as src:
            a, t = src.read(1).astype(np.float32), src.transform
            if src.nodata is not None:
                a[a == src.nodata] = np.nan
        c0, r0 = int(round((t.c - e0) / DEM_RES)), int(round((n1 - t.f) / DEM_RES))
        rs, cs = slice(max(r0, 0), min(r0 + a.shape[0], h)), slice(max(c0, 0), min(c0 + a.shape[1], w))
        if rs.start < rs.stop and cs.start < cs.stop:
            z[rs, cs] = a[rs.start - r0:rs.stop - r0, cs.start - c0:cs.stop - c0]

    with ThreadPoolExecutor(8) as ex:
        list(ex.map(place, [href for _, href in latest.values()]))
    missing = np.isnan(z)
    log(f"DEM mosaic {z.shape}, missing share {missing.mean():.4f}")
    assert missing.mean() < 0.05, "DEM mosaic has holes"
    if missing.any():
        z = z[tuple(ndimage.distance_transform_edt(missing, return_distances=False, return_indices=True))]
    r = np.clip(((n1 - en[:, 1]) / DEM_RES).astype(int), 0, h - 1)
    c = np.clip(((en[:, 0] - e0) / DEM_RES).astype(int), 0, w - 1)
    out = {"dem_elev": z[r, c]}
    for scale in (6, 20, 60):
        zs = ndimage.gaussian_filter(z, scale / DEM_RES / 2)
        gy, gx = np.gradient(zs, DEM_RES)
        out[f"dem_slope_{scale}m"] = np.degrees(np.arctan(np.hypot(gx, gy)))[r, c]
        out[f"dem_curv_{scale}m"] = (ndimage.laplace(zs) / DEM_RES ** 2)[r, c]
        if scale == 20:
            norm = np.hypot(gx, gy)[r, c] + 1e-9
            out["dem_north_20m"], out["dem_east_20m"] = gy[r, c] / norm, -gx[r, c] / norm
    for radius in (50, 150, 500):
        out[f"dem_tpi_{radius}m"] = z[r, c] - ndimage.uniform_filter(z, int(2 * radius / DEM_RES) + 1)[r, c]
    return pd.DataFrame(out)


def sample_raster(href, en):
    import rasterio
    from pyproj import Transformer
    with rasterio.open("/vsicurl/" + href) as src:
        xy = np.column_stack(Transformer.from_crs(2056, src.crs, always_xy=True).transform(en[:, 0], en[:, 1]))
        vals = np.array([v[0] for v in src.sample([tuple(p) for p in xy])], float)
        if src.nodata is not None:
            vals[vals == src.nodata] = np.nan
    return vals


def has_identify(layer, en):
    try:
        return layer_table([layer], en[:3]).shape[1] > 0
    except Exception as e:
        log(f"{layer}: identify unavailable ({type(e).__name__})")
        return False


def extra_layers(en):
    found = []
    for text in EXTRA_LAYER_SEARCHES:
        res = fetch_json(f"{API}/api/SearchServer?searchText={urllib.parse.quote(text)}&type=layers&lang=de")["results"]
        found += [r["attrs"]["layer"] for r in res if "layer" in r.get("attrs", {})]
    found = sorted(set(found))
    log(f"extra layers found by search: {found}")
    cols, partitions = {}, set()
    for layer in found:
        try:
            if has_identify(layer, en):
                t = layer_table([layer], en)
                kept = []
                for k in t.columns:
                    v = pd.to_numeric(t[k], errors="coerce")
                    codes = pd.factorize(t[k].astype(str))[0]
                    if codes.max() < 1 or tuple(codes) in partitions:
                        continue
                    partitions.add(tuple(codes))
                    cols[f"ext_{k}"] = v.to_numpy() if v.notna().mean() > 0.9 else codes
                    kept.append(k)
                log(f"{layer}: identify ok, kept {kept} (columns repeating an earlier partition dropped)")
                continue
            tif = [a["href"] for it in stac_items(layer, wgs_bbox(en)) for a in it["assets"].values()
                   if a["href"].endswith((".tif", ".tiff"))]
            if not tif:
                log(f"{layer}: no identify attributes and no STAC GeoTIFF -> skipped")
                continue
            v = sample_raster(tif[0], en)
            if np.isfinite(v).mean() > 0.9 and np.nanstd(v) > 0:
                cols[f"ext_{layer.split('.')[-1]}"] = v
            log(f"{layer}: STAC GeoTIFF {tif[0].rsplit('/', 1)[-1]}, finite {np.isfinite(v).mean():.3f}, "
                f"value quantiles 0/50/100 % {np.round(np.nanquantile(v, [0, .5, 1]), 3).tolist()}, "
                f"distinct {len(np.unique(v[np.isfinite(v)]))}")
        except Exception as e:
            log(f"{layer}: failed ({type(e).__name__}: {str(e)[:200]}) -> skipped")
    return pd.DataFrame(cols)


def bbox4(v):
    return [v["xmin"], v["ymin"], v["xmax"], v["ymax"]] if isinstance(v, dict) else list(v)[:4]


def aef_tiles(lonlat):
    st = pd.read_parquet(io.BytesIO(fetch(AEF_INDEX)))
    log(f"AEF STAC index: {len(st)} rows, columns {[(c, str(st[c].dtype)) for c in st.columns]}")
    bb = np.array([bbox4(v) for v in st["bbox"]], float)
    lo, hi = lonlat.min(0), lonlat.max(0)
    hit = np.flatnonzero((bb[:, 0] <= hi[0]) & (bb[:, 2] >= lo[0]) & (bb[:, 1] <= hi[1]) & (bb[:, 3] >= lo[1]))
    log(f"AEF rows intersecting the site extent lon {lo[0]:.4f}-{hi[0]:.4f} lat {lo[1]:.4f}-{hi[1]:.4f}: {len(hit)}")
    if not len(hit):
        utm32 = st["proj:epsg"].astype(str).str.contains("32632").to_numpy()
        log(f"example bbox values of EPSG:32632 rows: {bb[utm32][:3].tolist()}; raw {st['bbox'].iloc[0]}")
        raise RuntimeError("no AEF tile intersects the sites")
    log(f"first intersecting row: {json.dumps(st.iloc[hit[0]].drop(labels=['geometry']).to_dict(), default=str)[:1500]}")
    tiles = []
    for k in hit:
        m = re.search(r"v1/annual/(\d{4})/\w+/[\w.-]+?\.tiff?", json.dumps(st.iloc[k].drop(labels=["geometry"]).to_dict(), default=str))
        if m and int(m.group(1)) in AEF_YEARS:
            tiles.append((int(m.group(1)), AEF_BASE + m.group(0), bb[k]))
    assert tiles, "no tile path found in the intersecting AEF rows"
    return sorted(tiles, key=lambda t: (t[0], t[1]))


def aef_window(url, lonlat):
    import rasterio
    from pyproj import Transformer
    with rasterio.open("/vsicurl/" + url) as src:
        xy = np.column_stack(Transformer.from_crs(4326, src.crs, always_xy=True).transform(lonlat[:, 0], lonlat[:, 1]))
        col, rw = ~src.transform * (xy[:, 0], xy[:, 1])
        col, rw = np.floor(col).astype(int), np.floor(rw).astype(int)
        c0, r0 = col.min() - 1, rw.min() - 1
        raw = src.read(window=rasterio.windows.Window(c0, r0, col.max() - c0 + 2, rw.max() - r0 + 2)).astype(np.float32)
    de = np.sign(raw) * (raw / 127.5) ** 2
    de[:, raw[0] == -128] = np.nan
    return np.stack([np.nanmean(de[:, rr - r0 - 1:rr - r0 + 2, cc - c0 - 1:cc - c0 + 2].reshape(64, -1), 1)
                     for rr, cc in zip(rw, col)])


def aef_embeddings(lonlat):
    years = {}
    for year, url, box in aef_tiles(lonlat):
        v = years.setdefault(year, np.full((len(lonlat), 64), np.nan))
        todo = np.flatnonzero(np.isnan(v[:, 0]) & (lonlat[:, 0] >= box[0]) & (lonlat[:, 0] <= box[2])
                              & (lonlat[:, 1] >= box[1]) & (lonlat[:, 1] <= box[3]))
        if len(todo):
            v[todo] = aef_window(url, lonlat[todo])
            log(f"AEF {year} {url.rsplit('/', 1)[-1]}: {len(todo)} sites, finite {np.isfinite(v[todo]).all(1).mean():.3f}")
    log(f"AEF years: {sorted(years)}; sites covered per year {[int(np.isfinite(v).all(1).sum()) for v in years.values()]}")
    emb = np.nanmean(np.stack(list(years.values())), 0)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    assert np.isfinite(emb).all(), "AEF embeddings have gaps"
    return pd.DataFrame(emb, columns=[f"aef_{i:02d}" for i in range(64)])


def block(name, fn):
    try:
        out = fn()
        log(f"block {name}: {out.shape[1]} columns")
        return out
    except Exception as e:
        log(f"block {name}: FAILED ({type(e).__name__}: {str(e)[:300]}) -> left out")
        return pd.DataFrame()


if __name__ == "__main__":
    train, test = load()
    pyreadr, path, cols = jura_rda()
    g = jura_grid(pyreadr, path)
    extra = transect(pyreadr, path, cols, train, test, g)
    path.unlink()
    parts = [train.assign(split="train"), test.assign(split="test")]
    if extra is not None:
        parts.append(extra.assign(split="transect"))
    sites = pd.concat(parts, ignore_index=True)
    assert sites.loc[sites["split"] == "test", [m for m in METALS if m in sites]].isna().all().all()
    en = to_lv95(g)(sites)
    sites["E"], sites["N"] = en[:, 0], en[:, 1]
    log(f"sites: {sites['split'].value_counts().to_dict()}; LV95 E {en[:, 0].min():.0f}-{en[:, 0].max():.0f}, "
        f"N {en[:, 1].min():.0f}-{en[:, 1].max():.0f}")

    feats = [sites, neighbourhood(sites, g),
             block("geology", lambda: layer_table(GEOLOGY_LAYERS, en)),
             block("dem", lambda: dem_terrain(en)),
             block("extra layers", lambda: extra_layers(en)),
             block("aef", lambda: aef_embeddings(lv95_to_wgs(en)))]
    out = pd.concat(feats, axis=1)
    assert len(out) == len(sites)
    out.to_csv(OUT / OUT_FILE, index=False)
    log(f"wrote {OUT_FILE}: {out.shape}")
    tr = out[out["split"] != "test"]
    num = [c for c in out.columns if c.startswith(("dem_", "ext_", "rock", "dist_")) and pd.api.types.is_numeric_dtype(out[c])]
    log(f"Spearman with Co (train + transect rows): {tr[num].corrwith(tr['Co'], method='spearman').round(3).sort_values().to_dict()}")
    log(f"done; download {OUT / OUT_FILE}")
