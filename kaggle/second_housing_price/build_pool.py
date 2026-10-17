import glob
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from build_external import (CENTER, DISTRICT, EARTH_M, SALE_RANGE, Lookup, district_en, load_sources, lookup_column,
                            norm, num, tree)

LEVEL_ZH = {"低楼层": "low", "中楼层": "mid", "高楼层": "high", "顶层": "top", "底层": "ground", "地下室": "basement",
            "低层": "low", "中层": "mid", "高层": "high", "上叠": "upper_duplex", "下叠": "ground"}
RENO_ZH = [("毛坯", "bare"), ("精", "premium"), ("豪华", "premium"), ("简", "basic"), ("其他", "other")]
STRUCT_ZH = {"板楼": "slab", "塔楼": "tower", "板塔结合": "mixed", "平房": "bungalow", "暂无数据": "no_data"}
FAMILY_SUFFIX = re.compile(r"((甲|乙)?[\dA-Za-z一二三四五六七八九十]+(号院|号楼|号|期|区|院|街区|组团)|[东西南北中](区|院|里)|"
                           r"(一|二|三|四|五|六|七|八|九|十)期|[A-Za-z]座)$")
POOL_COLS = ["source", "row_key", "pool_comm", "pool_family", "pool_biz", "district", "area", "bedrooms", "living_rooms", "is_room_layout",
             "floor_level", "floor_total", "year_built", "faces_south", "faces_north", "through", "n_directions",
             "renovation", "structure", "near_subway", "lat", "lng", "coord_level", "dist_center_km", "ring_zone",
             "dist_subway_m", "n_subway_1km", "price"]


def family(pool_comm):
    dist, name = pool_comm.split("|", 1)
    prev = None
    while prev != name:
        prev = name
        stripped = FAMILY_SUFFIX.sub("", name)
        if len(stripped) >= 2:
            name = stripped
    return f"{dist}|{name}"


def parse_layout(s):
    m = s.astype(str).str.extract(r"(\d+)(室|房间)(\d+)厅")
    return pd.to_numeric(m[0]), pd.to_numeric(m[2]), (m[1] == "房间").astype(float).where(m[0].notna())


def parse_floor(s):
    s = s.astype(str).str.replace(" ", "", regex=False)
    m = s.str.extract(r"^(\D+?)[(/（]?共(\d+)层")
    level = m[0].str.strip("(/（").map(LEVEL_ZH)
    plain = s.str.extract(r"^(\d+)层$")[0]
    total = pd.to_numeric(m[1]).fillna(pd.to_numeric(plain))
    level = level.where(level.notna(), np.where(plain.notna(), "unstated", None))
    return level, total


def parse_orientation(s):
    s = s.fillna("").astype(str)
    south = s.str.contains("南")
    north = s.str.contains("北")
    n = s.str.replace(r"[^东南西北 ]", "", regex=True).str.split().map(len)
    n = n.where(s.str.contains(" "), s.str.count(r"[东西南北]").clip(upper=4))
    return south.astype(float), north.astype(float), (south & north).astype(float), n.astype(float)


def parse_reno(s):
    s = s.fillna("").astype(str)
    out = pd.Series(np.nan, index=s.index, dtype=object)
    for key, val in RENO_ZH:
        out = out.where(out.notna() | ~s.str.contains(key), val)
    return out


def harmonise(source, comm, dist, biz, area, layout, floor, orient, reno, struct, year, subway, price):
    bed, liv, room = parse_layout(layout)
    level, total = parse_floor(floor) if floor is not None else (pd.Series(np.nan, index=comm.index), np.nan)
    south, north, through, ndir = parse_orientation(orient) if orient is not None else (np.nan,) * 4
    df = pd.DataFrame({
        "source": source, "pool_comm": dist.fillna("?") + "|" + comm.map(norm), "pool_biz": biz.map(norm) if biz is not None else "",
        "district": dist, "area": pd.to_numeric(area, errors="coerce"), "bedrooms": bed, "living_rooms": liv,
        "is_room_layout": room, "floor_level": level, "floor_total": total, "year_built": year,
        "faces_south": south, "faces_north": north, "through": through, "n_directions": ndir,
        "renovation": parse_reno(reno) if reno is not None else np.nan,
        "structure": struct.astype(str).str.strip().map(STRUCT_ZH) if struct is not None else np.nan,
        "near_subway": subway, "price": pd.to_numeric(price, errors="coerce"), "_comm": comm, "_dist": dist,
    })
    df = df[df["price"].between(*SALE_RANGE) & (df["area"] > 10) & (comm.map(norm) != "")]
    return df.drop_duplicates(["pool_comm", "area", "price", "floor_level", "floor_total"])


def external_rows(g, street_to_district):
    out = []
    z = pd.concat([pd.read_excel(f) for f in sorted(glob.glob(str(g / "zhangzhenzhe" / "*.xlsx")))])
    z = z.drop_duplicates(["房源", "房子信息"]).reset_index(drop=True)
    parts = z["所在区域"].astype(str).str.strip().str.split(r"\s+")
    info = z["房子信息"].astype(str)
    pick = lambda pat: info.str.split("|").map(lambda xs: next((x.strip() for x in xs if re.search(pat, x)), np.nan))
    out.append(harmonise("zhangzhenzhe", parts.str[0], parts.str[-1].map(street_to_district), parts.str[-1],
                         num(pick(r"平米")), pick(r"室|房间"), pick(r"层"), pick(r"^[东南西北 ]+$"), pick(r"精装|简装|毛坯|其他"),
                         pick(r"板楼|塔楼|板塔结合|平房|暂无数据"), pd.to_numeric(info.str.extract(r"(\d{4})年")[0]),
                         z["标签"].astype(str).str.contains("近地铁").astype(float), num(z["单价"])))
    m = pd.read_csv(g / "msq0313" / "beijing1211_origin_utf8.csv", on_bad_lines="skip", engine="python")
    out.append(harmonise("msq0313", m["小区名称"], district_en(m["所在区域"]), None, num(m["建筑面积"]), m["房屋户型"],
                         m["所在楼层"], m["房屋朝向"], m["装修情况"], m["建筑类型"], np.nan, np.nan, num(m["单价"])))
    x = pd.read_csv(g / "xiaoyusmd" / "lianjia.csv").drop_duplicates()
    out.append(harmonise("xiaoyusmd", x["Garden"], district_en(x["Region"]), x["District"], x["Size"], x["Layout"],
                         x["Floor"].astype("Int64").astype(str) + "层", x["Direction"], x["Renovation"], None,
                         pd.to_numeric(x["Year"], errors="coerce"), np.nan, x["Price"] * 1e4 / x["Size"]))
    o = pd.read_csv(g / "onceonceonce" / "data.csv", header=None,
                    names=["layout", "floor", "area", "struct_reno", "total", "comm", "district", "elevator", "date", "use", "price"])
    out.append(harmonise("onceonceonce", o["comm"], district_en(o["district"]), None, o["area"], o["layout"], o["floor"],
                         None, o["struct_reno"].astype(str).str.split("/").str[-1], None, np.nan, np.nan, o["price"]))
    a = pd.read_csv(g / "ariosolzq" / "7-test-file.csv", encoding="gbk")
    out.append(harmonise("ariosolzq", a["小区"], district_en(a["行政区"]), a["社区"], a["面积"], a["户型"], None,
                         a["朝向"], a["装修类型"], None, np.nan, np.nan, a["平方米价格"]))
    e = pd.read_csv(g / "echoetic" / "beijing_housing_clean_data.csv")
    out.append(harmonise("echoetic", e["小区名称"], district_en(e["城区"]), e["区域"], e["面积_数值"], e["户型"], e["楼层"],
                         e["朝向"], e["装修"], None, pd.to_numeric(e["建筑年份"], errors="coerce"),
                         e["标签"].astype(str).str.contains("近地铁").astype(float), e["单价_数值"]))
    return pd.concat(out, ignore_index=True)


def competition_rows(df, split):
    lay = df["layout"].fillna("").str.extract(r"^(\d+)(bed|room)(\d+)living$")
    fl = df["floor"].fillna("").str.extract(r"^([a-z_]+)\((\d+)F total\)$")
    fx = df["floor"].fillna("").str.extract(r"^F(\d+)$")[0]
    level = fl[0].where(fl[0].notna(), np.where(fx.notna(), "unstated", None))
    tokens = df["orientation"].fillna("").str.split()
    south = tokens.map(lambda t: any("S" in x for x in t))
    north = tokens.map(lambda t: any("N" in x for x in t))
    return pd.DataFrame({
        "source": "competition", "row_key": split + ":" + df["row_id"].astype(str),
        "pool_comm": df["district"] + "|" + df["community_zh"].map(norm), "pool_biz": df["street_zh"].map(norm),
        "district": df["district"], "area": df["area_sqm"], "bedrooms": pd.to_numeric(lay[0]),
        "living_rooms": pd.to_numeric(lay[2]), "is_room_layout": (lay[1] == "room").astype(float).where(lay[0].notna()),
        "floor_level": level, "floor_total": pd.to_numeric(fl[1]).fillna(pd.to_numeric(fx)),
        "year_built": df["year_built"].where(df["year_built"] <= 2024), "faces_south": south.astype(float),
        "faces_north": north.astype(float), "through": (south & north).astype(float), "n_directions": tokens.map(len).astype(float),
        "renovation": df["renovation"].where(df["renovation"].isin(["premium", "basic", "bare", "other"])),
        "structure": df["structure"], "near_subway": df["tag"].eq("near_subway").astype(float), "price": np.nan,
        "_comm": df["community_zh"], "_dist": df["district"],
    })


def main(data_dir):
    data_dir = Path(data_dir)
    tr, te = pd.read_csv(data_dir / "train.csv"), pd.read_csv(data_dir / "test.csv")
    ours = pd.concat([tr, te])
    street_to_district = ours.drop_duplicates("street_zh").set_index("street_zh")["district"]
    _, _, coords, biz_chrom, stations, _, _, _, ring_hulls = load_sources(data_dir / "external", street_to_district)

    ext = external_rows(data_dir / "external" / "github", street_to_district)
    comp = pd.concat([competition_rows(tr, "train"), competition_rows(te, "test")], ignore_index=True)
    pool = pd.concat([ext, comp], ignore_index=True)
    pool["row_key"] = pool["row_key"].fillna(pool["source"] + ":" + pool.index.astype(str))

    keys = list(zip(pool["_dist"].fillna("?"), pool["_comm"].map(norm)))
    uniq = list(dict.fromkeys(keys))
    lat = np.full(len(uniq), np.nan)
    lng = np.full(len(uniq), np.nan)
    lvl = np.full(len(uniq), np.nan)
    for level in sorted(coords, reverse=True):
        lk = Lookup(coords[level], cols=("lat", "lng"))
        la, ln = lookup_column(uniq, lk, "lat"), lookup_column(uniq, lk, "lng")
        fill = np.isnan(lat) & ~np.isnan(la)
        lat[fill], lng[fill], lvl[fill] = la[fill], ln[fill], level
    pos = {k: i for i, k in enumerate(uniq)}
    idx = np.array([pos[k] for k in keys])
    pool["lat"], pool["lng"], pool["coord_level"] = lat[idx], lng[idx], lvl[idx]

    named = pool["pool_biz"].fillna("") != ""
    biz_c = pool[named & (pool["coord_level"] >= 3)].groupby("pool_biz")[["lat", "lng"]].median()
    chrom = biz_chrom.assign(k=biz_chrom["comm"].map(norm)).drop_duplicates("k").set_index("k")[["lat", "lng"]]
    biz_c = biz_c.combine_first(chrom)
    b_lat = pool["pool_biz"].map(biz_c["lat"]).where(named)
    b_lng = pool["pool_biz"].map(biz_c["lng"]).where(named)
    wide = pool["pool_biz"].fillna("").str.endswith("其他")
    far = (np.hypot((pool["lat"] - b_lat) * 111.0, (pool["lng"] - b_lng) * 85.0) > 6.0) & b_lat.notna() & ~wide
    pool.loc[far, ["lat", "lng", "coord_level"]] = np.nan
    miss = pool["lat"].isna() & b_lat.notna()
    pool.loc[miss, "lat"], pool.loc[miss, "lng"], pool.loc[miss, "coord_level"] = b_lat[miss], b_lng[miss], 0
    dist_c = pool[pool["coord_level"] >= 3].groupby("district")[["lat", "lng"]].median()
    miss = pool["lat"].isna() & pool["district"].isin(dist_c.index)
    pool.loc[miss, "lat"] = pool.loc[miss, "district"].map(dist_c["lat"])
    pool.loc[miss, "lng"] = pool.loc[miss, "district"].map(dist_c["lng"])
    pool.loc[miss, "coord_level"] = -1
    print("coordinates dropped as >6 km from their business-area centre:", int(far.sum()))

    has = pool["lat"].notna().to_numpy()
    q = np.radians(pool.loc[has, ["lat", "lng"]].to_numpy())
    pool.loc[has, "dist_center_km"] = tree([CENTER[0]], [CENTER[1]]).query(q, k=1)[0][:, 0] * EARTH_M / 1000
    st = tree(stations["lat"], stations["lng"])
    pool.loc[has, "dist_subway_m"] = st.query(q, k=1)[0][:, 0] * EARTH_M
    pool.loc[has, "n_subway_1km"] = st.query_radius(q, r=1000 / EARTH_M, count_only=True)
    pts = pool.loc[has, ["lat", "lng"]].to_numpy()
    zone = np.full(len(pts), 7.0)
    for n in sorted(ring_hulls, reverse=True):
        zone[ring_hulls[n].find_simplex(pts) >= 0] = n
    pool.loc[has, "ring_zone"] = zone

    pool["pool_family"] = pool["pool_comm"].map(family)
    pool = pool[POOL_COLS]
    assert pool.loc[pool["source"] == "competition", "price"].isna().all()
    assert pool["row_key"].is_unique
    pool.to_csv(data_dir / "ext_pool.csv", index=False)
    ext_part = pool[pool["source"] != "competition"]
    print("ext_pool", pool.shape, "external rows by source:", ext_part["source"].value_counts().to_dict())
    print("external coverage:", {c: round(float(ext_part[c].notna().mean()), 2) for c in
                                 ["bedrooms", "floor_level", "floor_total", "year_built", "renovation", "structure",
                                  "faces_south", "near_subway", "lat", "pool_biz"]})
    print("coord_level external:", ext_part["coord_level"].value_counts(dropna=False).sort_index().to_dict())
    comp_part = pool[pool["source"] == "competition"]
    shared = set(comp_part["pool_comm"]) & set(ext_part["pool_comm"])
    print("competition rows whose community is in the external pool:", round(float(comp_part["pool_comm"].isin(shared).mean()), 3))
    fam = set(ext_part["pool_family"])
    print("... whose community or community family is in the pool:",
          round(float((comp_part["pool_comm"].isin(shared) | comp_part["pool_family"].isin(fam)).mean()), 3))
    print("unmapped district values in external rows:", int(ext_part["district"].isna().sum()), "| known districts:", len(set(DISTRICT.values())))


if __name__ == "__main__":
    main(sys.argv[1])
