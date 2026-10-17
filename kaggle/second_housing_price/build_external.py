import glob
import json
import math
import re
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
from sklearn.neighbors import BallTree

CENTER = (39.9087, 116.3975)
EARTH_M = 6371000.0
REFERENCE_SOURCE = "ariosolzq"
SALE_RANGE = (5000, 250000)
RENT_RANGE = (10, 600)
DISTRICT = {
    "朝阳": "Chaoyang", "丰台": "Fengtai", "海淀": "Haidian", "通州": "Tongzhou", "昌平": "Changping",
    "大兴": "Daxing", "顺义": "Shunyi", "房山": "Fangshan", "东城": "Dongcheng", "西城": "Xicheng",
    "石景山": "Shijingshan", "门头沟": "Mentougou", "密云": "Miyun", "亦庄开发区": "Yizhuang",
    "北京经济技术开发区": "Yizhuang", "怀柔": "Huairou", "平谷": "Pinggu", "延庆": "Yanqing",
}
DISTRICT_ALTERNATIVES = {"Yizhuang": ["Yizhuang", "Daxing"], "Daxing": ["Daxing", "Yizhuang"]}
TOP_SCHOOL = (r"附属|附小|附中|实验|史家|北京小学|中关村第|育才|景山|北京四中|北京八中|十一学校|人大附|清华附|北大附|北师大|"
              r"汇文|广渠门|府学")
POI_DIST = ["school", "top_school", "university", "hospital", "park", "mall"]
POI_COUNT = {"school": 1000, "kindergarten": 1000, "office": 1000, "supermarket": 1000, "university": 2000,
             "hospital": 2000, "park": 2000, "top_school": 2000}
COMP_LOG_AREA = 0.25
TXN_COMP_RADII = (200, 500)
TXN_COMP_MIN = 3


def norm(s):
    if pd.isna(s):
        return ""
    s = re.sub(r"[\s·\-—_()（）\[\]【】]", "", str(s))
    return re.sub(r"小区$", "", s)


def district_en(s):
    s = s.astype("string").str.strip().str.replace(r"区$", "", regex=True)
    return s.map(lambda v: DISTRICT.get(v, DISTRICT.get(f"{v}区")) if pd.notna(v) else None)


def num(s):
    return pd.to_numeric(s.astype(str).str.replace(",", "").str.extract(r"(\d+\.?\d*)")[0], errors="coerce")


def _tlat(x, y):
    r = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * np.sqrt(np.abs(x))
    r += (20.0 * np.sin(6.0 * x * math.pi) + 20.0 * np.sin(2.0 * x * math.pi)) * 2.0 / 3.0
    r += (20.0 * np.sin(y * math.pi) + 40.0 * np.sin(y / 3.0 * math.pi)) * 2.0 / 3.0
    r += (160.0 * np.sin(y / 12.0 * math.pi) + 320.0 * np.sin(y * math.pi / 30.0)) * 2.0 / 3.0
    return r


def _tlng(x, y):
    r = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * np.sqrt(np.abs(x))
    r += (20.0 * np.sin(6.0 * x * math.pi) + 20.0 * np.sin(2.0 * x * math.pi)) * 2.0 / 3.0
    r += (20.0 * np.sin(x * math.pi) + 40.0 * np.sin(x / 3.0 * math.pi)) * 2.0 / 3.0
    r += (150.0 * np.sin(x / 12.0 * math.pi) + 300.0 * np.sin(x / 30.0 * math.pi)) * 2.0 / 3.0
    return r


def wgs_to_gcj(lng, lat):
    a, ee = 6378245.0, 0.00669342162296594323
    dlat, dlng = _tlat(lng - 105.0, lat - 35.0), _tlng(lng - 105.0, lat - 35.0)
    rad = lat / 180.0 * math.pi
    magic = 1 - ee * np.sin(rad) ** 2
    sq = np.sqrt(magic)
    dlat = dlat * 180.0 / ((a * (1 - ee)) / (magic * sq) * math.pi)
    dlng = dlng * 180.0 / (a / sq * np.cos(rad) * math.pi)
    return lng + dlng, lat + dlat


def gcj_to_wgs(lng, lat):
    lng, lat = np.asarray(lng, float), np.asarray(lat, float)
    w_lng, w_lat = lng.copy(), lat.copy()
    for _ in range(5):
        g_lng, g_lat = wgs_to_gcj(w_lng, w_lat)
        w_lng, w_lat = w_lng - (g_lng - lng), w_lat - (g_lat - lat)
    return w_lng, w_lat


def bd_to_wgs(lng, lat):
    x, y = np.asarray(lng, float) - 0.0065, np.asarray(lat, float) - 0.006
    x_pi = math.pi * 3000.0 / 180.0
    z = np.sqrt(x * x + y * y) - 0.00002 * np.sin(y * x_pi)
    theta = np.arctan2(y, x) - 0.000003 * np.cos(x * x_pi)
    return gcj_to_wgs(z * np.cos(theta), z * np.sin(theta))


def frame(comm, dist, biz, value, **extra):
    n = len(comm)
    as_series = lambda v: pd.Series(v, index=comm.index) if np.ndim(v) else pd.Series([v] * n, index=comm.index)
    out = pd.DataFrame({"comm": comm, "dist": as_series(dist), "biz": as_series(biz), "value": as_series(value)})
    for k, v in extra.items():
        out[k] = as_series(v)
    return out.reset_index(drop=True)


def load_sources(ext, street_to_district):
    g = ext / "github"
    sale, rent = {}, {}

    a = pd.read_csv(g / "ariosolzq" / "7-test-file.csv", encoding="gbk")
    sale["ariosolzq"] = frame(a["小区"], district_en(a["行政区"]), a["社区"], a["平方米价格"])
    z = pd.concat([pd.read_excel(f) for f in sorted(glob.glob(str(g / "zhangzhenzhe" / "*.xlsx")))])
    z = z.drop_duplicates(["房源", "房子信息"]).reset_index(drop=True)
    parts = z["所在区域"].astype(str).str.strip().str.split(r"\s+")
    sale["zhangzhenzhe"] = frame(parts.str[0], parts.str[-1].map(street_to_district), parts.str[-1], num(z["单价"]))
    x = pd.read_csv(g / "xiaoyusmd" / "lianjia.csv")
    sale["xiaoyusmd"] = frame(x["Garden"], district_en(x["Region"]), x["District"], x["Price"] * 1e4 / x["Size"])
    m = pd.read_csv(g / "msq0313" / "beijing1211_origin_utf8.csv", on_bad_lines="skip", engine="python")
    sale["msq0313"] = frame(m["小区名称"], district_en(m["所在区域"]), np.nan, num(m["单价"]))
    e = pd.read_csv(g / "echoetic" / "beijing_housing_clean_data.csv")
    sale["echoetic"] = frame(e["小区名称"], district_en(e["城区"]), e["区域"], e["单价_数值"])
    o = pd.read_csv(g / "onceonceonce" / "data.csv", header=None,
                    names=["layout", "floor", "area", "struct_reno", "total", "comm", "district", "elevator", "date", "use", "price"])
    sale["onceonceonce"] = frame(o["comm"], district_en(o["district"]), np.nan, pd.to_numeric(o["price"], errors="coerce"))
    z_area = num(z["房子信息"].astype(str).str.split("|").str[1])
    listings = {
        "ariosolzq": frame(a["小区"], district_en(a["行政区"]), np.nan, a["平方米价格"], area=a["面积"]),
        "zhangzhenzhe": frame(parts.str[0], parts.str[-1].map(street_to_district), np.nan, num(z["单价"]), area=z_area),
        "xiaoyusmd": frame(x["Garden"], district_en(x["Region"]), np.nan, x["Price"] * 1e4 / x["Size"], area=x["Size"]),
        "msq0313": frame(m["小区名称"], district_en(m["所在区域"]), np.nan, num(m["单价"]), area=num(m["建筑面积"])),
        "onceonceonce": frame(o["comm"], district_en(o["district"]), np.nan, pd.to_numeric(o["price"], errors="coerce"),
                              area=pd.to_numeric(o["area"], errors="coerce")),
    }

    con = sqlite3.connect(g / "chrom826" / "beijingmap.sqlite")
    pc = pd.read_sql("select * from pricebycommunity", con)
    hb = pd.read_sql("select * from houseprice", con)
    con.close()
    pc_lng, pc_lat = bd_to_wgs(pd.to_numeric(pc["lng"], errors="coerce"), pd.to_numeric(pc["lat"], errors="coerce"))
    sale["chrom826"] = frame(pc["name"], np.nan, np.nan, pd.to_numeric(pc["price"], errors="coerce"))
    hb_lng, hb_lat = bd_to_wgs(pd.to_numeric(hb["lng"], errors="coerce"), pd.to_numeric(hb["lat"], errors="coerce"))
    biz_chrom = frame(hb["name"], np.nan, hb["name"], pd.to_numeric(hb["price"], errors="coerce"), lng=hb_lng, lat=hb_lat)

    c = pd.read_csv(g / "caleyuexing" / "beijing_lianjia.csv")
    rent["caleyuexing"] = frame(c["地址"], district_en(c["区域"]), c["板块"], c["总价"] / c["面积"])
    y = pd.read_csv(g / "yokumii" / "bj_rental_with_coordinate.csv")
    y_dist = district_en(y["区域"])
    rent["yokumii"] = frame(y["小区"], y_dist, y["板块"], pd.to_numeric(y["单位租金"], errors="coerce"))

    precise = (y["坐标精度"].isna() | (y["坐标精度"] == "小区")).to_numpy()
    y_lng, y_lat = gcj_to_wgs(pd.to_numeric(y["经度"], errors="coerce"), pd.to_numeric(y["纬度"], errors="coerce"))
    osm = json.load(open(ext / "osm" / "named_places.json", encoding="utf-8"))
    rows = []
    for el in osm["elements"]:
        t = el.get("tags", {})
        cc = el.get("center") or {"lat": el.get("lat"), "lon": el.get("lon")}
        if t.get("name") and cc.get("lat") is not None:
            rows.append({"comm": t["name"], "lng": cc["lon"], "lat": cc["lat"], "res": t.get("landuse") == "residential"})
    osm = pd.DataFrame(rows)
    coords = {
        4: frame(y["小区"][precise], y_dist[precise], np.nan, 0.0, lng=y_lng[precise], lat=y_lat[precise]),
        3: frame(pc["name"], np.nan, np.nan, 0.0, lng=pc_lng, lat=pc_lat),
        2: frame(osm["comm"][osm["res"]], np.nan, np.nan, 0.0, lng=osm["lng"][osm["res"]].to_numpy(), lat=osm["lat"][osm["res"]].to_numpy()),
        1: frame(osm["comm"][~osm["res"]], np.nan, np.nan, 0.0, lng=osm["lng"][~osm["res"]].to_numpy(), lat=osm["lat"][~osm["res"]].to_numpy()),
    }

    st = json.load(open(ext / "osm" / "subway_stations.json", encoding="utf-8"))
    stations = pd.DataFrame([{"lat": (el.get("center") or el)["lat"], "lng": (el.get("center") or el)["lon"],
                              "name": el.get("tags", {}).get("name")} for el in st["elements"]]).drop_duplicates("name")

    txn = pd.concat([pd.read_csv(g / "yasante7" / f"housing_data{yr}.csv", low_memory=False,
                                 usecols=["Lon", "Lat", "price_per_sq", "square"]) for yr in (2015, 2016, 2017)])
    txn = txn[txn["price_per_sq"].between(*SALE_RANGE) & txn["Lon"].notna()]
    t_lng, t_lat = bd_to_wgs(txn["Lon"], txn["Lat"])
    txn = pd.DataFrame({"lng": t_lng, "lat": t_lat, "price": txn["price_per_sq"].to_numpy(),
                        "area": pd.to_numeric(txn["square"], errors="coerce").to_numpy()})
    pois = json.load(open(ext / "osm" / "pois.json", encoding="utf-8"))
    prow = []
    for el in pois["elements"]:
        t = el.get("tags", {})
        cc = el.get("center") or {"lat": el.get("lat"), "lon": el.get("lon")}
        if cc.get("lat") is None:
            continue
        am, name = t.get("amenity"), t.get("name", "")
        kinds = []
        if am == "school":
            kinds.append("school")
            if re.search(TOP_SCHOOL, name):
                kinds.append("top_school")
        elif am == "kindergarten":
            kinds.append("kindergarten")
        elif am in ("university", "college"):
            kinds.append("university")
        elif am == "hospital":
            kinds.append("hospital")
        elif t.get("leisure") == "park":
            kinds.append("park")
        elif t.get("shop") in ("mall", "department_store"):
            kinds.append("mall")
        elif t.get("shop") == "supermarket":
            kinds.append("supermarket")
        elif "office" in t:
            kinds.append("office")
        for k in kinds:
            prow.append({"kind": k, "lat": cc["lat"], "lng": cc["lon"]})
    pois = pd.DataFrame(prow)

    rings = json.load(open(ext / "osm" / "ring_roads.json", encoding="utf-8"))
    ring_pts = {}
    for el in rings["elements"]:
        hit = re.search(r"([二三四五六])环", el["tags"].get("name", ""))
        if hit:
            n = "二三四五六".index(hit.group(1)) + 2
            ring_pts.setdefault(n, []).extend((p["lat"], p["lon"]) for p in el.get("geometry", []))
    ring_hulls = {n: Delaunay(np.array(pts)) for n, pts in ring_pts.items()}
    return sale, rent, coords, biz_chrom, stations, txn, listings, pois, ring_hulls


class Lookup:
    def __init__(self, df, value_range=None, cols=("value",)):
        d = df.assign(k=df["comm"].map(norm))
        d = d[d["k"] != ""]
        if value_range is not None:
            d = d[d["value"].between(*value_range)]
        cols = list(cols)
        self.by_dk = d[d["dist"].notna()].groupby(["dist", "k"])[cols].median()
        n_dist = d.groupby("k")["dist"].nunique()
        unique = n_dist.index[n_dist <= 1]
        self.by_k = d[d["k"].isin(unique)].groupby("k")[cols].median()

    def get(self, district, k):
        for dist in DISTRICT_ALTERNATIVES.get(district, [district]):
            if (dist, k) in self.by_dk.index:
                return self.by_dk.loc[(dist, k)]
        if k in self.by_k.index:
            return self.by_k.loc[k]
        return None


def lookup_column(keys, lk, col="value"):
    out = []
    for dist, k in keys:
        r = lk.get(dist, k)
        out.append(np.nan if r is None else float(r[col]))
    return np.array(out)


def align_offsets(columns):
    ref = np.log(columns[REFERENCE_SOURCE])
    offsets = {}
    for s, v in columns.items():
        both = ~np.isnan(ref) & ~np.isnan(v)
        offsets[s] = float(np.median(ref[both] - np.log(v[both]))) if both.sum() >= 20 else np.nan
    return offsets


def combine(columns, offsets):
    logs = np.column_stack([np.log(columns[s]) + offsets[s] for s in columns if not np.isnan(offsets[s])])
    n = (~np.isnan(logs)).sum(axis=1)
    with np.errstate(invalid="ignore"):
        mean = np.where(n > 0, np.nanmean(np.where(np.isnan(logs), np.nan, logs), axis=1), np.nan)
    return np.exp(mean), n


def comparables(rows, listings, offsets):
    frames = []
    for s, df in listings.items():
        if np.isnan(offsets.get(s, np.nan)):
            continue
        d = df[df["value"].between(*SALE_RANGE) & (df["area"] > 10)].copy()
        d["k"] = d["comm"].map(norm)
        d["logp"] = np.log(d["value"]) + offsets[s]
        frames.append(d[["k", "dist", "area", "logp"]])
    allx = pd.concat(frames, ignore_index=True)
    allx = allx[allx["k"] != ""]
    by_dk = {key: g for key, g in allx[allx["dist"].notna()].groupby(["dist", "k"])}
    n_dist = allx.groupby("k")["dist"].nunique()
    unique = set(n_dist.index[n_dist <= 1])
    by_k = {key: g for key, g in allx[allx["k"].isin(unique)].groupby("k")}
    px, n = [], []
    for comm_zh, dist, area in rows[["community_zh", "district", "area_sqm"]].itertuples(index=False):
        k = norm(comm_zh)
        cand = None
        for d in DISTRICT_ALTERNATIVES.get(dist, [dist]):
            if (d, k) in by_dk:
                cand = by_dk[(d, k)]
                break
        if cand is None:
            cand = by_k.get(k)
        if cand is None or not area > 0:
            px.append(np.nan)
            n.append(0)
            continue
        close = cand[np.abs(np.log(cand["area"] / area)) < COMP_LOG_AREA]
        px.append(float(np.exp(close["logp"].median())) if len(close) else np.nan)
        n.append(len(close))
    return pd.DataFrame({"community_zh": rows["community_zh"].to_numpy(), "area_sqm": rows["area_sqm"].to_numpy(),
                         "px_comp": px, "n_comp": n})


def txn_comparables(rows, txn, tx_tree):
    out = pd.DataFrame(index=rows.index)
    ok = (rows["coord_level"] >= 2).to_numpy() & (rows["area_sqm"] > 0).to_numpy()
    q = np.radians(rows.loc[ok, ["lat", "lng"]].to_numpy())
    areas = rows.loc[ok, "area_sqm"].to_numpy()
    logp, t_area = np.log(txn["price"].to_numpy()), txn["area"].to_numpy()
    for radius in TXN_COMP_RADII:
        px, n = np.full(len(rows), np.nan), np.zeros(len(rows))
        near = tx_tree.query_radius(q, r=radius / EARTH_M)
        pos = np.where(ok)[0]
        for j, idx in enumerate(near):
            close = idx[np.abs(np.log(t_area[idx] / areas[j])) < COMP_LOG_AREA] if len(idx) else idx
            if len(close) >= TXN_COMP_MIN:
                px[pos[j]], n[pos[j]] = np.exp(np.median(logp[close])), len(close)
        out[f"px_txn_comp_{radius}m"], out[f"n_txn_comp_{radius}m"] = px, n
    return out


def tree(lat, lng):
    return BallTree(np.radians(np.c_[lat, lng]), metric="haversine")


def main(data_dir):
    data_dir = Path(data_dir)
    tr = pd.read_csv(data_dir / "train.csv")
    te = pd.read_csv(data_dir / "test.csv")
    ours = pd.concat([tr, te])[["community_zh", "street_zh", "district"]]
    ours_rows = pd.concat([tr, te])[["community_zh", "district", "area_sqm"]].drop_duplicates(["community_zh", "area_sqm"])
    street_to_district = ours.drop_duplicates("street_zh").set_index("street_zh")["district"]
    sale, rent, coords, biz_chrom, stations, txn, listings, pois, ring_hulls = load_sources(data_dir / "external", street_to_district)

    comm = ours.drop_duplicates("community_zh").reset_index(drop=True)
    keys = list(zip(comm["district"], comm["community_zh"].map(norm)))
    px = {s: lookup_column(keys, Lookup(df, SALE_RANGE)) for s, df in sale.items()}
    for s, v in px.items():
        comm[f"px_{s}"] = v
    for s, df in rent.items():
        comm[f"rent_{s}"] = lookup_column(keys, Lookup(df, RENT_RANGE))
    offsets = align_offsets(px)
    print("sale-source log offsets to", REFERENCE_SOURCE, {s: round(o, 3) for s, o in offsets.items()})
    comm["px_combined"], comm["px_n_sources"] = combine(px, offsets)

    comm["lat"], comm["lng"], comm["coord_level"] = np.nan, np.nan, np.nan
    for level in sorted(coords, reverse=True):
        lk = Lookup(coords[level], cols=("lat", "lng"))
        todo = comm["lat"].isna().to_numpy()
        lat = lookup_column(keys, lk, "lat")
        lng = lookup_column(keys, lk, "lng")
        fill = todo & ~np.isnan(lat)
        comm.loc[fill, "lat"], comm.loc[fill, "lng"], comm.loc[fill, "coord_level"] = lat[fill], lng[fill], level

    street_c = comm[comm["coord_level"] >= 3].groupby("street_zh")[["lat", "lng"]].median()
    chrom_c = biz_chrom.drop_duplicates("comm").set_index("comm")[["lat", "lng"]]
    street_c = street_c.combine_first(chrom_c)
    s_lat, s_lng = comm["street_zh"].map(street_c["lat"]), comm["street_zh"].map(street_c["lng"])
    wide = comm["street_zh"].str.endswith("其他")
    far = (np.hypot((comm["lat"] - s_lat) * 111.0, (comm["lng"] - s_lng) * 85.0) > 6.0) & s_lat.notna() & ~wide
    print("coordinates dropped for being >6 km from their street centre:", int(far.sum()),
          comm.loc[far, "coord_level"].value_counts().to_dict())
    comm.loc[far, ["lat", "lng", "coord_level"]] = np.nan
    miss = comm["lat"].isna() & s_lat.notna()
    comm.loc[miss, "lat"], comm.loc[miss, "lng"], comm.loc[miss, "coord_level"] = s_lat[miss], s_lng[miss], 0

    has = comm["lat"].notna().to_numpy()
    q = np.radians(comm.loc[has, ["lat", "lng"]].to_numpy())
    comm.loc[has, "dist_center_km"] = tree([CENTER[0]], [CENTER[1]]).query(q, k=1)[0][:, 0] * EARTH_M / 1000
    st_tree = tree(stations["lat"], stations["lng"])
    comm.loc[has, "dist_subway_m"] = st_tree.query(q, k=1)[0][:, 0] * EARTH_M
    comm.loc[has, "n_subway_1km"] = st_tree.query_radius(q, r=1000 / EARTH_M, count_only=True)
    comm.loc[has, "n_subway_2km"] = st_tree.query_radius(q, r=2000 / EARTH_M, count_only=True)
    tx_tree = tree(txn["lat"], txn["lng"])
    prices = txn["price"].to_numpy()
    dist, idx = tx_tree.query(q, k=30)
    comm.loc[has, "txn_px_k30"] = np.median(prices[idx], axis=1)
    comm.loc[has, "txn_dist_k30_m"] = dist[:, -1] * EARTH_M
    near = tx_tree.query_radius(q, r=300 / EARTH_M)
    comm.loc[has, "txn_n_300m"] = [len(i) for i in near]
    comm.loc[has, "txn_px_300m"] = [np.median(prices[i]) if len(i) else np.nan for i in near]

    for kind in sorted(set(POI_DIST) | set(POI_COUNT)):
        pk = pois[pois["kind"] == kind]
        pt = tree(pk["lat"], pk["lng"])
        if kind in POI_DIST:
            comm.loc[has, f"dist_{kind}_m"] = pt.query(q, k=1)[0][:, 0] * EARTH_M
        if kind in POI_COUNT:
            comm.loc[has, f"n_{kind}_{POI_COUNT[kind] // 1000}km"] = pt.query_radius(q, r=POI_COUNT[kind] / EARTH_M, count_only=True)
    pts = comm.loc[has, ["lat", "lng"]].to_numpy()
    zone = np.full(len(pts), 7.0)
    for n in sorted(ring_hulls, reverse=True):
        zone[ring_hulls[n].find_simplex(pts) >= 0] = n
    comm.loc[has, "ring_zone"] = zone
    print("ring_zone counts:", comm["ring_zone"].value_counts().sort_index().to_dict())

    comps = comparables(ours_rows, listings, offsets)
    located = ours_rows.merge(comm[["community_zh", "lat", "lng", "coord_level"]], on="community_zh", how="left")
    comps = pd.concat([comps, txn_comparables(located, txn, tx_tree).reset_index(drop=True)], axis=1)
    print("txn comparables coverage:", {r: round(float(comps[f"px_txn_comp_{r}m"].notna().mean()), 3) for r in TXN_COMP_RADII})
    comps.to_csv(data_dir / "ext_comparable.csv", index=False)
    print("ext_comparable", comps.shape, "px_comp coverage", round(float(comps["px_comp"].notna().mean()), 3))

    street = ours.drop_duplicates("street_zh")[["street_zh", "district"]].reset_index(drop=True)
    skeys = list(zip(street["district"], street["street_zh"].map(norm)))
    biz_px = {}
    for s, df in list(sale.items()) + [("chrom826", biz_chrom)]:
        if df["biz"].notna().any():
            biz_px[s] = lookup_column(skeys, Lookup(df.assign(comm=df["biz"]), SALE_RANGE))
            street[f"spx_{s}"] = biz_px[s]
    for s, df in rent.items():
        street[f"srent_{s}"] = lookup_column(skeys, Lookup(df.assign(comm=df["biz"]), RENT_RANGE))
    street["spx_combined"], street["spx_n_sources"] = combine(biz_px, {s: offsets[s] for s in biz_px})

    comm = comm.drop(columns=["street_zh", "district"])
    street = street.drop(columns=["district"])
    comm.to_csv(data_dir / "ext_community.csv", index=False)
    street.to_csv(data_dir / "ext_street.csv", index=False)
    print("ext_community", comm.shape, "ext_street", street.shape)
    print("community coverage:", {c: round(float(comm[c].notna().mean()), 3) for c in
                                  ["px_combined", "rent_yokumii", "lat", "txn_px_k30"]})
    print("coord_level counts:", comm["coord_level"].value_counts(dropna=False).sort_index().to_dict())
    print("street coverage spx_combined:", round(float(street["spx_combined"].notna().mean()), 3))


if __name__ == "__main__":
    main(sys.argv[1])
