import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
EXT = DATA / "external"
LAT0, LON0 = 51.50735, -0.12776
KX, KY = 111.32 * np.cos(np.radians(LAT0)), 111.19
BOX = (51.25, 51.72, -0.56, 0.36)
GN_CODES = {"PPLX", "PPL", "PPLL", "PPLA3", "PPLA4", "ST", "PRK", "MUS", "STDM", "BDG", "TOWR", "HSTS", "SQR",
            "MKT", "CSTL", "PAL", "MNMT", "THTR", "ZOO", "GDN", "AMTH", "UNIV", "RECG", "HLL"}
SPREAD_KM = 4.0
SUFFIX = r"(?:station|tube|underground|overground|dlr|rail|stop)"
STOPWORDS = {"bank", "angel", "temple", "oval", "borough", "city", "london", "central", "west", "east", "north",
             "south", "park", "green", "common", "village", "the", "garden", "gardens", "court", "cross", "church",
             "hill", "station", "river", "canal", "market", "square", "high street", "town", "centre", "center",
             "north end", "east end", "west end", "the city", "oxford", "cambridge", "brighton", "kent", "surrey",
             "essex", "wales", "scotland", "england", "hampton", "victoria", "chelsea", "arsenal", "the mall",
             "the strand", "strand", "mill", "manor", "abbey", "palace", "beckton", "ivy", "lea", "cray", "deptford"}
LINES = r"(?:line|lines)"
FAR_KM = 2.5
ITERS = 3


def norm(s):
    s = re.sub(r"<[^>]+>", " ", s.lower())
    s = re.sub(r"['’`.]", "", s)
    s = re.sub(r"\bst\b", "saint", s)
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def xy(lat, lon):
    return (np.asarray(lon) - LON0) * KX, (np.asarray(lat) - LAT0) * KY


def gazetteer():
    st = pd.read_csv(EXT / "doogal" / "london_stations.csv")
    st["zone"] = st.Zone.astype(str).str.extract(r"(\d+)")[0].astype(float)
    st["key"] = st.Station.str.replace(r"\s*\(.*\)", "", regex=True).map(norm)
    cols = "id name asciiname alt lat lon fclass fcode cc cc2 a1 a2 a3 a4 pop elev dem tz mod".split()
    gn = pd.read_csv(EXT / "geonames" / "GB.txt", sep="\t", names=cols, quoting=3, low_memory=False)
    gn = gn[gn.lat.between(*BOX[:2]) & gn.lon.between(*BOX[2:]) & gn.fcode.isin(GN_CODES)]
    gn = gn.assign(key=gn.asciiname.map(norm))
    g = pd.concat([st[["key", "Latitude", "Longitude"]].set_axis(["key", "lat", "lon"], axis=1).assign(station=1),
                   gn[["key", "lat", "lon"]].assign(station=0)], ignore_index=True)
    g = g[(g.key.str.len() >= 4) & (~g.key.isin(STOPWORDS) | (g.station == 1))]
    g["x"], g["y"] = xy(g.lat, g.lon)
    agg = g.groupby("key").agg(x=("x", "median"), y=("y", "median"), sx=("x", "std"), sy=("y", "std"),
                               station=("station", "max")).fillna(0)
    agg = agg[np.hypot(agg.sx, agg.sy) < SPREAD_KM]
    agg["suffix_only"] = agg.index.isin(STOPWORDS)
    print(f"gazetteer: {len(agg)} names ({int(agg.station.sum())} stations) from {len(g)} entries")
    return agg, st


def mentions(text, names, suffix_only, maxlen):
    w = text.split()
    out, i = [], 0
    while i < len(w):
        for n in range(min(maxlen, len(w) - i), 0, -1):
            k = " ".join(w[i:i + n])
            if k in names:
                after = " ".join(w[i + n:i + n + 2])
                if re.match(LINES, after):
                    break
                has_suffix = bool(re.match(SUFFIX, after))
                if k in suffix_only and not has_suffix:
                    break
                out.append((k, 2.0 if has_suffix else 1.0))
                i += n - 1
                break
        i += 1
    return out


def geomedian(p, w, it=50):
    m = np.average(p, axis=0, weights=w)
    for _ in range(it):
        d = np.maximum(np.hypot(*(p - m).T), 1e-3)
        m = np.average(p, axis=0, weights=w / d)
    return m


def build():
    agg, st = gazetteer()
    names, suffix_only = set(agg.index), set(agg.index[agg.suffix_only])
    maxlen = max(len(k.split()) for k in names)
    df = pd.concat([pd.read_csv(DATA / f) for f in ["train.csv", "test.csv"]], ignore_index=True)
    rows = []
    for cell, name, desc in zip(df.geo_cell, df.name, df.description.fillna("")):
        for field, wt in ((name, 3.0), (desc, 1.0)):
            for k, w in mentions(norm(field), names, suffix_only, maxlen):
                rows.append((cell, k, w * wt))
    m = pd.DataFrame(rows, columns=["cell", "key", "w"]).groupby(["cell", "key"], as_index=False).w.sum()
    m["x"], m["y"] = agg.x.reindex(m.key).values, agg.y.reindex(m.key).values
    print(f"mentions: {len(m)} (cell, name) pairs over {m.cell.nunique()} of {df.geo_cell.nunique()} cells")

    spec = pd.Series(1.0, index=agg.index)
    for it in range(ITERS):
        m["ew"] = m.w * spec.reindex(m.key).values
        est = {}
        for cell, g in m.groupby("cell"):
            p, w = g[["x", "y"]].values, g.ew.values
            c = geomedian(p, w)
            keep = np.hypot(*(p - c).T) < FAR_KM
            if keep.any():
                c = geomedian(p[keep], w[keep])
            est[cell] = c
        e = pd.DataFrame(est, index=["ex", "ey"]).T
        d = np.hypot(m.x - e.ex.reindex(m.cell).values, m.y - e.ey.reindex(m.cell).values)
        med = pd.Series(d.values, index=m.key).groupby(level=0).median()
        spec = (1.0 / (1.0 + (med / 1.5) ** 2)).reindex(agg.index).fillna(1.0)
        print(f"iter {it}: median mention distance to its cell estimate {np.median(d):.2f} km")

    m["d"] = d.values
    m["near"] = m.d < 1.0
    q = m.assign(wn=m.ew * m.near).groupby("cell").agg(w=("ew", "sum"), wn=("wn", "sum"), n_names=("key", "size"))
    out = e.join(q)
    out["geo_conf"] = out.wn / out.w
    out["geo_w"] = np.log1p(out.w)
    sx, sy = xy(st.Latitude, st.Longitude)
    ds = np.hypot(out.ex.values[:, None] - sx[None], out.ey.values[:, None] - sy[None])
    out["station_km"] = ds.min(axis=1)
    out["station_zone"] = st.zone.values[ds.argmin(axis=1)]
    out["stations_1km"] = (ds < 1.0).sum(axis=1)
    out["centre_km"] = np.hypot(out.ex, out.ey)
    out = out.rename(columns={"ex": "geo_x", "ey": "geo_y"}).drop(columns=["w", "wn"])
    out.index.name = "geo_cell"
    out = out.reset_index()
    split_half_check(m)
    out.to_csv(DATA / "ext_geo.csv", index=False)
    print(f"wrote ext_geo.csv: {len(out)} cells; geo_conf median {out.geo_conf.median():.2f}; "
          f"centre_km quartiles {out.centre_km.quantile([.25, .5, .75]).round(1).tolist()}")


def split_half_check(m):
    rng = np.random.default_rng(0)
    gaps = []
    for _, g in m.groupby("cell"):
        if len(g) < 6:
            continue
        mask = rng.random(len(g)) < 0.5
        if mask.sum() < 3 or (~mask).sum() < 3:
            continue
        a = geomedian(g[["x", "y"]].values[mask], g.ew.values[mask])
        b = geomedian(g[["x", "y"]].values[~mask], g.ew.values[~mask])
        gaps.append(np.hypot(*(a - b)))
    gaps = np.array(gaps)
    print(f"split-half: {len(gaps)} cells, distance between half-estimates median {np.median(gaps):.2f} km, "
          f"p75 {np.quantile(gaps, .75):.2f}, p90 {np.quantile(gaps, .9):.2f}")


if __name__ == "__main__":
    build()
