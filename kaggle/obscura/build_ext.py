import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EXT = ROOT / "data" / "external"
FAC_CSV = EXT / "facilities.csv"
DATE0, DATE1 = "2019-01-01", "2022-01-01"
CAPS = {"power_coal": 1600, "power_gas": 1600, "power_oil": 200, "refinery": 500, "metal": 450, "chemical": 350,
        "gas_plant": 300, "flare": 1200, "oilfield": 450, "cement": 120, "coal_mine": 200}
CHIP_M = 3840
BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]
CLOUDY = [0, 1, 3, 8, 9, 10]
GAS = {"ch4": ("COPERNICUS/S5P/OFFL/L3_CH4", "CH4_column_volume_mixing_ratio_dry_air_bias_corrected"),
       "no2": ("COPERNICUS/S5P/OFFL/L3_NO2", "tropospheric_NO2_column_number_density"),
       "co": ("COPERNICUS/S5P/OFFL/L3_CO", "CO_column_number_density")}
AAI = ("COPERNICUS/S5P/OFFL/L3_AER_AI", "absorbing_aerosol_index")
S5P_SCALE = 1113.2


def osm_kind(t):
    ind, prod = (t.get("industrial") or "").lower(), (t.get("product") or "").lower()
    if t.get("man_made") == "flare":
        return "flare"
    if ind == "refinery" or "petrol" in prod:
        return "refinery"
    if ind in ("steelmaker", "steel", "smelter", "aluminium_smelter") or any(k in prod for k in ("steel", "aluminium", "copper")):
        return "metal"
    if ind in ("chemical", "petrochemical") or any(k in prod for k in ("chemical", "fertili")):
        return "chemical"
    if ind in ("gas", "liquefied_natural_gas", "lng"):
        return "gas_plant"
    if ind == "oil" or prod == "oil":
        return "oil_point"
    if "cement" in prod:
        return "cement"
    if ind == "mine":
        return "coal_mine"
    return None


def facilities():
    g = pd.read_csv(EXT / "gppd" / "gppd.csv", low_memory=False)
    g = g[g.primary_fuel.isin(["Coal", "Gas", "Oil"]) & (g.capacity_mw >= 300)]
    rows = [("power_" + f.lower(), la, lo) for f, la, lo in zip(g.primary_fuel, g.latitude, g.longitude)]
    try:
        elements = json.load(open(EXT / "osm" / "osm.json"))["elements"]
    except (OSError, ValueError, KeyError) as err:
        print(f"OSM skipped ({type(err).__name__}): power plants only")
        elements = []
    for e in elements:
        la, lo = e.get("lat", e.get("center", {}).get("lat")), e.get("lon", e.get("center", {}).get("lon"))
        k = osm_kind(e.get("tags", {}))
        if la is not None and k:
            rows.append((k, la, lo))
    d = pd.DataFrame(rows, columns=["kind", "lat", "lon"])
    d["cell"] = (d.lat // 0.1).astype(int).astype(str) + "_" + (d.lon // 0.1).astype(int).astype(str)
    oil = d[d.kind == "oil_point"].groupby("cell").agg(lat=("lat", "mean"), lon=("lon", "mean"), n=("lat", "size"))
    oil = oil[oil.n >= 20].reset_index().assign(kind="oilfield")[["kind", "lat", "lon", "cell"]]
    flare = d[d.kind == "flare"].groupby("cell").agg(lat=("lat", "mean"), lon=("lon", "mean")).reset_index().assign(kind="flare")
    rest = d[~d.kind.isin(["oil_point", "flare"])].drop_duplicates(["kind", "cell"])
    d = pd.concat([rest, oil, flare[["kind", "lat", "lon", "cell"]]], ignore_index=True)
    d = d[(d.lat.abs() < 70)].sample(frac=1, random_state=0).drop_duplicates("cell")
    old = pd.read_csv(FAC_CSV) if FAC_CSV.exists() else pd.DataFrame(columns=["fac_id", "kind", "lat", "lon", "cell"])
    d = d[~d.cell.isin(old.cell)]
    have = old.kind.value_counts()
    d = pd.concat([v.head(max(CAPS[k] - have.get(k, 0), 0)) for k, v in d.groupby("kind") if k in CAPS], ignore_index=True)
    d.insert(0, "fac_id", np.arange(len(d)) + (int(old.fac_id.max()) + 1 if len(old) else 0))
    d = pd.concat([old, d], ignore_index=True)
    d.to_csv(FAC_CSV, index=False)
    print(f"added {len(d) - len(old)}:", d.kind.value_counts().to_dict(), "total", len(d))


V3_KINDS = ["flare_viirs", "mine"]
FLARE_CELLS = EXT / "eog" / "flare_cells_selected.csv"
MINES = EXT / "mines_gt2km2.csv"


def cell_of(lat, lon):
    return (lat // 0.1).astype(int).astype(str) + "_" + (lon // 0.1).astype(int).astype(str)


def facilities_v3(n_mines=1200):
    old = pd.read_csv(FAC_CSV)
    fl = pd.read_csv(FLARE_CELLS)
    fl = fl[fl.land == 1].assign(kind="flare_viirs")[["kind", "lat", "lon", "cell"]]
    mi = pd.read_csv(MINES).dropna(subset=["elev"])
    mi = mi[mi.land == 1].assign(kind="mine", cell=lambda d: cell_of(d.lat, d.lon)).drop_duplicates("cell")
    mi = mi[~mi.cell.isin(old.cell)].sample(frac=1, random_state=0)
    mi = pd.concat([mi[mi.elev > 700].head(n_mines // 2), mi[mi.elev <= 700].head(n_mines // 2)])[["kind", "lat", "lon", "cell"]]
    d = pd.concat([fl[~fl.cell.isin(old.cell)], mi], ignore_index=True).drop_duplicates("cell")
    d.insert(0, "fac_id", np.arange(len(d)) + int(old.fac_id.max()) + 1)
    pd.concat([old, d], ignore_index=True).to_csv(FAC_CSV, index=False)
    print(f"added {len(d)}:", d.kind.value_counts().to_dict(), "total", len(old) + len(d))


def ee_sample_fn(ee):
    s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    era_h = ee.ImageCollection("ECMWF/ERA5_LAND/HOURLY")
    era_d = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
    dem = ee.ImageCollection("COPERNICUS/DEM/GLO30").select("DEM").mosaic().setDefaultProjection("EPSG:4326", None, 30)
    slope = ee.Terrain.slope(dem)
    land = ee.Image("MODIS/006/MOD44W/2015_01_01").select("water_mask").eq(0)
    firms = ee.ImageCollection("FIRMS").select("T21")
    pct = ee.Reducer.percentile([5, 50, 95, 99.9], ["p5", "p50", "p95", "p999"], None, None, 200000)

    def window_mean(c, band):
        return ee.Image(ee.Algorithms.If(c.size().gt(0), c.mean(), ee.Image.constant(0).rename(band).updateMask(0)))

    def per_sample(f):
        pt = f.geometry()
        t = ee.Date(f.get("date"))
        img = ee.Image(s2.filterBounds(pt).filterDate(t, t.advance(1, "day")).first())
        chip = pt.buffer(CHIP_M / 2).bounds()
        x = img.select(BANDS).toFloat()
        scl, cld = img.select("SCL"), img.select("MSK_CLDPRB")
        valid = x.select("B4").gt(0)
        nocloud = valid.And(scl.remap(CLOUDY, [0] * len(CLOUDY), 1))
        clear = nocloud.And(cld.lt(50))
        rr = dict(geometry=chip, scale=10, maxPixels=1e8)
        n_clear = clear.selfMask().reduceRegion(ee.Reducer.count(), **rr).values().get(0)
        use = ee.Image(ee.Algorithms.If(ee.Number(n_clear).gte(500), clear, valid))
        st = x.updateMask(use).reduceRegion(pct, **rr)
        sclf = scl.eq(ee.Image.constant(list(range(12)))).rename([f"scl{c}" for c in range(12)]).reduceRegion(ee.Reducer.mean(), **rr)
        misc = ee.Image.cat(clear.rename("clear_frac"), cld.rename("cld_mean"), x.eq(0).reduce("mean").rename("zero_frac"),
                            cld.gte(40).rename("s2_cloud_frac")).reduceRegion(ee.Reducer.mean(), **rr)
        centre = pt.buffer(CHIP_M / 6)
        ctr = x.select(["B4", "B8", "B11", "B12"]).updateMask(use).reduceRegion(ee.Reducer.median(), geometry=centre, scale=10, maxPixels=1e8)
        b8a, b11, b12 = x.select("B8A"), x.select("B11"), x.select("B12")
        cand = nocloud.And(b12.gt(1500))
        tai = b12.subtract(b11).divide(b8a.max(1)).rename("tai")
        hot = cand.And(tai.gte(0.45)).And(b12.subtract(b11).gte(b11.subtract(b8a)))
        hs = ee.Image.cat(hot.rename("hot_n"), nocloud.And(b12.gt(6000)).rename("b12_hi_n")).reduceRegion(ee.Reducer.sum(), **rr)
        ts = tai.updateMask(cand).reduceRegion(ee.Reducer.max().combine(ee.Reducer.percentile([99.9], ["p999"], None, None, 200000), "", True), **rr)

        utc_h = ee.Number(13.5).subtract(pt.coordinates().getNumber(0).divide(15)).round()
        hours = era_h.filterDate(t.advance(utc_h.subtract(1), "hour"), t.advance(utc_h.add(2), "hour"))
        wind = hours.map(lambda h: h.expression("sqrt(u*u+v*v)", {"u": h.select("u_component_of_wind_10m"),
                                                                  "v": h.select("v_component_of_wind_10m")}).rename("ws")).mean()
        uv = hours.select(["u_component_of_wind_10m", "v_component_of_wind_10m"]).mean()
        day = era_d.filterDate(t, t.advance(1, "day")).first()
        era = ee.Image.cat(wind, uv, day.select(["temperature_2m", "dewpoint_temperature_2m", "total_precipitation_sum",
                                                 "surface_solar_radiation_downwards_sum"]))
        wx = era.reduceRegion(ee.Reducer.first(), geometry=pt, scale=11132)
        topo = ee.Image.cat(dem.rename("elevation"), slope.rename("slope")).reduceRegion(ee.Reducer.mean(), geometry=chip, scale=30, maxPixels=1e8)

        site, ring = pt.buffer(5000).bounds(), pt.buffer(75000).difference(pt.buffer(30000), 1000)
        w0, w1 = t.advance(-1, "day"), t.advance(2, "day")
        tg = {}
        for name, (coll, band) in GAS.items():
            im = window_mean(ee.ImageCollection(coll).select(band).filterDate(w0, w1), band).updateMask(land)
            s = im.reduceRegion(ee.Reducer.mean().combine(ee.Reducer.count(), "", True), geometry=site, scale=S5P_SCALE)
            r = im.reduceRegion(ee.Reducer.median().combine(ee.Reducer.count(), "", True), geometry=ring, scale=S5P_SCALE, maxPixels=1e8)
            tg.update({f"y_{name}_site": s.get(f"{band}_mean"), f"n_{name}_site": s.get(f"{band}_count"),
                       f"y_{name}_ring": r.get(f"{band}_median"), f"n_{name}_ring": r.get(f"{band}_count")})
        a = window_mean(ee.ImageCollection(AAI[0]).select(AAI[1]).filterDate(w0, w1), AAI[1]).updateMask(land).reduceRegion(
            ee.Reducer.mean().combine(ee.Reducer.count(), "", True), geometry=site, scale=S5P_SCALE)
        fr = firms.filterDate(w0, w1).max().reduceRegion(ee.Reducer.max(), geometry=site, scale=1000)
        tg.update({"y_aai": a.get(f"{AAI[1]}_mean"), "n_aai": a.get(f"{AAI[1]}_count"), "y_firms_T21": fr.get("T21")})
        ctr = ctr.rename(ctr.keys(), ctr.keys().map(lambda k: ee.String(k).cat("_ctr")))
        props = ee.Dictionary(tg).combine(st).combine(sclf).combine(misc).combine(hs).combine(ts).combine(wx).combine(topo).combine(ctr)
        return ee.Feature(None, props.combine({"fac_id": f.get("fac_id"), "date": f.get("date"), "s2_id": img.get("system:index")}))
    return per_sample


def pick_dates(ee, fac, seed, n):
    s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")

    def per_fac(f):
        ims = s2.filterBounds(f.geometry()).filterDate(DATE0, DATE1).filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 20)) \
            .randomColumn("r", seed).sort("r").limit(n)
        return ims.map(lambda im: ee.Feature(f.geometry(), {"fac_id": f.get("fac_id"), "date": ee.Image(im).date().format("YYYY-MM-dd")}))
    feats = [ee.Feature(ee.Geometry.Point([r.lon, r.lat]), {"fac_id": int(r.fac_id)}) for r in fac.itertuples()]
    return ee.FeatureCollection(feats).map(per_fac).flatten()


EXTRA_KINDS = ["flare", "refinery", "gas_plant", "chemical", "metal", "oilfield"]


def rows_path(npass):
    return EXT / ("ext_rows.jsonl" if npass == 0 else f"ext_rows_p{npass}.jsonl")


def run_batch(ee, fn, batch, npass=0):
    samples = pick_dates(ee, batch, seed=int(batch.fac_id.iloc[0]) + 100003 * npass, n=2 if npass == 0 else 4)
    return [f["properties"] for f in samples.map(fn).getInfo()["features"]]


def extract(project, test=False, npass=0):
    import ee
    ee.Initialize(project=project)
    fac = pd.read_csv(FAC_CSV)
    fn = ee_sample_fn(ee)
    if test:
        t0 = time.time()
        rows = run_batch(ee, fn, fac.sample(2, random_state=1))
        print(f"{len(rows)} rows in {time.time() - t0:.0f}s")
        for r in rows:
            print(json.dumps(r, sort_keys=True)[:3000])
        return
    rows_file = rows_path(npass)
    if npass == 1:
        fac = fac[fac.kind.isin(EXTRA_KINDS)]
    elif npass == 2:
        fac = fac[fac.kind.isin(V3_KINDS) | fac.cell.isin(pd.read_csv(FLARE_CELLS).query("land == 1").cell)]
    done = {json.loads(line)["fac_id"] for line in open(rows_file)} if rows_file.exists() else set()
    todo = fac[~fac.fac_id.isin(done)]
    print(f"facilities {len(fac)}, done {len(done)}, todo {len(todo)}")

    def run(batch):
        for attempt in range(4):
            try:
                return run_batch(ee, fn, batch, npass)
            except Exception as err:
                print(f"  batch {batch.fac_id.iloc[0]} attempt {attempt}: {type(err).__name__}: {str(err)[:200]}")
                time.sleep(10 * (attempt + 1))
        return []

    batches = [todo.iloc[i:i + 10] for i in range(0, len(todo), 10)]
    t0 = time.time()
    with ThreadPoolExecutor(6) as pool:
        for n, rows in enumerate(pool.map(run, batches), 1):
            with open(rows_file, "a") as fh:
                fh.writelines(json.dumps(r) + "\n" for r in rows)
            if n % 10 == 0:
                print(f"  {n}/{len(batches)} batches, {time.time() - t0:.0f}s", flush=True)


CHIP_PX = 96
CHIP_BANDS = BANDS + ["SCL", "B12max", "TAImax"]
SHARD = 500


def chip_request(ee, s2_id, lat, lon):
    img = ee.Image(f"COPERNICUS/S2_SR_HARMONIZED/{s2_id}")
    b8a, b11, b12 = img.select("B8A").toFloat(), img.select("B11").toFloat(), img.select("B12").toFloat()
    tai = b12.subtract(b11).divide(b8a.max(1)).where(b12.lte(1500), -1)
    mx = lambda im, name: im.reduceResolution(ee.Reducer.max(), maxPixels=1024).rename(name)
    out = ee.Image.cat(img.select(BANDS + ["SCL"]).toUint16(), mx(img.select("B12"), "B12max").toUint16(),
                       mx(tai.add(1).multiply(1000).max(0).min(65535), "TAImax").toUint16())
    r = 6378137.0
    half = CHIP_M / 2 / np.cos(np.radians(lat))
    x0 = r * np.radians(lon) - half
    y0 = r * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2)) + half
    px = 2 * half / CHIP_PX
    grid = {"dimensions": {"width": CHIP_PX, "height": CHIP_PX}, "crsCode": "EPSG:3857",
            "affineTransform": {"scaleX": px, "shearX": 0, "translateX": x0, "shearY": 0, "scaleY": -px, "translateY": y0}}
    return {"expression": out, "fileFormat": "NUMPY_NDARRAY", "grid": grid}


def chips(project, limit=None):
    import ee
    ee.Initialize(project=project)
    rows = pd.read_csv(EXT / "ext_rows.csv", usecols=["fac_id", "date", "s2_id"]).merge(
        pd.read_csv(FAC_CSV)[["fac_id", "lat", "lon"]], on="fac_id", how="left")
    rows = rows.head(limit) if limit else rows
    out = EXT / ("chips_pilot" if limit else "chips")
    out.mkdir(exist_ok=True)
    have = sorted(out.glob("shard_*.npz"))
    done = {(int(f), str(d)) for z in map(np.load, have) for f, d in zip(z["fac_id"], z["date"])}
    rows = rows[[(int(f), str(d)) not in done for f, d in zip(rows.fac_id, rows.date)]]
    first = int(have[-1].stem.split("_")[1]) + 1 if have else 0
    print(f"chips: {len(done)} done in {len(have)} shards, {len(rows)} to fetch")

    def one(r):
        for attempt in range(4):
            try:
                a = ee.data.computePixels(chip_request(ee, r.s2_id, r.lat, r.lon))
                return np.stack([a[b] for b in CHIP_BANDS])
            except Exception as err:
                print(f"  {r.fac_id} {r.date} attempt {attempt}: {type(err).__name__}: {str(err)[:150]}")
                time.sleep(5 * (attempt + 1))
        return np.zeros((len(CHIP_BANDS), CHIP_PX, CHIP_PX), np.uint16)

    t0 = time.time()
    for k in range(0, len(rows), SHARD):
        path = out / f"shard_{first + k // SHARD:04d}.npz"
        part = rows.iloc[k:k + SHARD]
        with ThreadPoolExecutor(8) as pool:
            x = np.stack(list(pool.map(one, part.itertuples())))
        np.savez_compressed(path, x=x, fac_id=part.fac_id.to_numpy(), date=part.date.to_numpy().astype(str))
        print(f"  shard {first + k // SHARD}: {len(part)} rows, failed {int((x.reshape(len(x), -1).max(1) == 0).sum())}, "
              f"{path.stat().st_size / 1e6:.1f} MB, {time.time() - t0:.0f}s", flush=True)


def pack_chips():
    import zipfile
    out = ROOT / "data" / "obscura-chips.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
        for p in sorted((EXT / "chips").glob("shard_*.npz")):
            z.write(p, p.name)
    print(f"-> {out} ({out.stat().st_size / 1e9:.2f} GB)")


def pack():
    import zipfile
    rows = pd.DataFrame([json.loads(line) for p in sorted(EXT.glob("ext_rows*.jsonl")) for line in open(p)])
    rows = rows.drop_duplicates(["fac_id", "date"]).merge(pd.read_csv(FAC_CSV)[["fac_id", "kind"]], on="fac_id", how="left")
    rows.to_csv(EXT / "ext_rows.csv", index=False)
    out = ROOT / "data" / "obscura-ext.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(EXT / "ext_rows.csv", "ext_rows.csv")
    print(f"{len(rows)} rows, {rows.fac_id.nunique()} facilities -> {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "facilities":
        facilities()
    elif cmd == "facilities_v3":
        facilities_v3()
    elif cmd == "pack":
        pack()
    elif cmd == "pack_chips":
        pack_chips()
    elif cmd == "chips":
        chips(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else None)
    else:
        extract(sys.argv[2], test=cmd == "test", npass=int(sys.argv[3]) if len(sys.argv) > 3 else 0)
