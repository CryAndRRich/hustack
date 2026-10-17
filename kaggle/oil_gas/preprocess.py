import re
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

TRAIN_FILE = "train_oil.csv"
TEST_FILE = "oil_test.csv"
TARGET = "Onshore/Offshore"
ID = "index"
GROUP = "Field name"

LABELS = {"OFFSHORE": 0, "ONSHORE": 1, "ONSHORE-OFFSHORE": 2}
CLASS_NAMES = ["OFFSHORE", "ONSHORE", "ONSHORE-OFFSHORE"]

NUMERIC = [
    "Latitude", "Longitude", "Depth",
    "Thickness (gross average ft)", "Thickness (net pay average ft)",
    "Porosity", "Permeability",
]
CATEGORICAL = [
    "Country", "Region", "Basin name", "Tectonic regime", "Operator company",
    "Hydrocarbon type", "Reservoir status", "Structural setting", "Reservoir period", "Lithology",
]
MULTI_VALUED = ["Tectonic regime", "Structural setting"]
MIN_TOKEN_COUNT = 5
MISSING = "NA"

GEO_RADII_KM = [1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120, 200]
GEO_DIRECTIONS = 16
GEO_FRAC_RADII_KM = [5, 20, 50, 120]
GEO_FAR_KM = 250.0
KM_PER_DEG = 111.32

EXT_COORDS_FILE = "ext_field_coords.csv"
EXT_TOPO_FILE = "ext_gebco_rings.csv"
EXT_COORDS_COLS = {"field", "ext_lat", "ext_lon", "ext_source"}
EXT_TOPO_COLS = {"pt_lat", "pt_lon", "radius_km", "bearing_deg", "elev_m"}
KEY_DECIMALS = 4
TOPO_NEAREST_KM = 2.0
TOPO_DISC_RADII_KM = [5, 30]
TOPO_RING_RADII_KM = [2, 8, 30]

GROUP_IMPUTE = [("Basin name", 1), ("Operator company", 3)]

DATA_DIR = Path("/kaggle/input/datasets/cryandrrich/oil-gas")
OUTPUT_DIR = Path("/kaggle/working")


def get_data_dir() -> Path:
    if not Path("/kaggle/input").exists():
        try:
            return Path(__file__).resolve().parent.parent / "data"
        except NameError:
            return Path.cwd() / "data"
    if not (DATA_DIR / TRAIN_FILE).exists() or not (DATA_DIR / TEST_FILE).exists():
        found = sorted(str(p) for p in DATA_DIR.rglob("*")) if DATA_DIR.exists() else "directory missing"
        raise FileNotFoundError(f"{TRAIN_FILE} / {TEST_FILE} not in {DATA_DIR}; found: {found}")
    return DATA_DIR


def get_output_dir() -> Path:
    return OUTPUT_DIR if OUTPUT_DIR.exists() else Path(".")


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = get_data_dir()
    print(f"data: {data_dir}")
    train = pd.read_csv(data_dir / TRAIN_FILE)
    test = pd.read_csv(data_dir / TEST_FILE)
    assert len(train) == 309 and len(test) == 133, (len(train), len(test))
    assert list(test.columns) == [c for c in train.columns if c != TARGET], "column mismatch"

    unknown = set(train[TARGET].dropna()) - set(LABELS)
    assert not unknown and train[TARGET].notna().all(), f"unexpected target values {unknown}"
    train["y"] = train[TARGET].map(LABELS).astype(int)
    return train, test


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in NUMERIC:
        raw = df[c]
        df[c] = pd.to_numeric(raw, errors="coerce")
        bad = int(df[c].isna().sum() - raw.isna().sum())
        if bad:
            print(f"  {c}: {bad} non-numeric values -> NaN")
    for c in CATEGORICAL + [GROUP]:
        df[c] = df[c].fillna(MISSING).astype(str).str.strip().str.upper()

    lone = df["Latitude"].isna() ^ df["Longitude"].isna()
    df.loc[lone, ["Latitude", "Longitude"]] = np.nan
    ok = df["Latitude"].between(-90, 90) & df["Longitude"].between(-180, 180)
    df.loc[~ok, ["Latitude", "Longitude"]] = np.nan

    df["coord_source"] = np.where(df["Latitude"].notna(), "orig", "none")
    return df


def impute_coords_by_field(train: pd.DataFrame, test: pd.DataFrame) -> None:

    both = pd.concat([train, test], ignore_index=True)
    known = both.dropna(subset=["Latitude", "Longitude"]).groupby(GROUP)[["Latitude", "Longitude"]].median()
    for name, df in (("train", train), ("test", test)):
        miss = df["Latitude"].isna()
        fill = df.loc[miss, GROUP].map(known["Latitude"]).notna()
        idx = fill[fill].index
        df.loc[idx, "Latitude"] = df.loc[idx, GROUP].map(known["Latitude"])
        df.loc[idx, "Longitude"] = df.loc[idx, GROUP].map(known["Longitude"])
        df.loc[idx, "coord_source"] = "field"
        print(f"  {name}: {int(miss.sum())} rows without coordinates, {len(idx)} filled from the same field")


def find_external(name: str) -> Path | None:

    base = get_data_dir()
    for cand in (base / name, base / "external" / name):
        if cand.exists():
            return cand
    if Path("/kaggle/input").exists():
        hits = sorted(Path("/kaggle/input").rglob(name))
        if hits:
            return hits[0]
    return None


def load_external() -> tuple[pd.DataFrame | None, pd.DataFrame | None]:

    out = []
    for name, need in ((EXT_COORDS_FILE, EXT_COORDS_COLS), (EXT_TOPO_FILE, EXT_TOPO_COLS)):
        path = find_external(name)
        if path is None:
            print(f"[ext] {name} NOT FOUND -- arms that need it are SKIPPED")
            out.append(None)
            continue
        d = pd.read_csv(path)
        lacking = need - set(d.columns)
        if lacking:
            print(f"[ext] {name} lacks columns {sorted(lacking)} -- arms that need it are SKIPPED")
            out.append(None)
            continue
        print(f"[ext] {name}: {len(d)} rows from {path}")
        out.append(d)
    return out[0], out[1]


def apply_external_coords(train: pd.DataFrame, test: pd.DataFrame, ext: pd.DataFrame) -> None:

    table = ext.dropna(subset=["ext_lat", "ext_lon"]).copy()
    table["field"] = table["field"].astype(str).str.strip().str.upper()
    table = table.drop_duplicates("field").set_index("field")
    for name, df in (("train", train), ("test", test)):
        miss = df["Latitude"].isna() | df["Longitude"].isna()
        la = df.loc[miss, GROUP].map(table["ext_lat"])
        lo = df.loc[miss, GROUP].map(table["ext_lon"])
        idx = la.index[la.notna() & lo.notna()]
        df.loc[idx, "Latitude"] = la[idx]
        df.loc[idx, "Longitude"] = lo[idx]
        df.loc[idx, "coord_source"] = "ext"
        names = sorted(df.loc[idx, GROUP].unique())
        print(f"  {name}: {int(miss.sum())} rows still without coordinates, {len(idx)} filled externally {names}")


def topo_table(rings: pd.DataFrame) -> pd.DataFrame:

    r = rings.copy()
    r["k_lat"] = r["pt_lat"].round(KEY_DECIMALS)
    r["k_lon"] = r["pt_lon"].round(KEY_DECIMALS)
    keys = ["k_lat", "k_lon"]
    feats = {"topo_elev_0": r[r["radius_km"] == 0].groupby(keys)["elev_m"].first()}
    for R in TOPO_DISC_RADII_KM:
        disc = r[r["radius_km"] <= R].groupby(keys)["elev_m"]
        feats[f"topo_min_{R}"] = disc.min()
        feats[f"topo_max_{R}"] = disc.max()
    for R in TOPO_RING_RADII_KM:
        ring = r[r["radius_km"] == R]
        feats[f"topo_below0_ring_{R}"] = (ring["elev_m"] < 0).groupby([ring["k_lat"], ring["k_lon"]]).mean()
    table = pd.DataFrame(feats)
    n_ring = r[r["radius_km"] > 0].groupby(keys).size()
    assert n_ring.nunique() == 1, f"uneven ring sample counts {n_ring.value_counts().to_dict()}"
    return table.astype(float)


def topo_features(df: pd.DataFrame, table: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:

    out = pd.DataFrame(np.nan, index=df.index, columns=table.columns)
    how = pd.Series("none", index=df.index)
    lat = df["Latitude"].round(KEY_DECIMALS)
    lon = df["Longitude"].round(KEY_DECIMALS)
    has = (lat.notna() & lon.notna()).to_numpy()
    c_lat = table.index.get_level_values(0).to_numpy(float)
    c_lon = table.index.get_level_values(1).to_numpy(float)
    for pos, i in enumerate(df.index):
        if not has[pos]:
            continue
        key = (lat.iat[pos], lon.iat[pos])
        if key in table.index:
            out.loc[i] = table.loc[key].to_numpy()
            how.loc[i] = "exact"
            continue
        d = haversine_km(key[0], key[1], c_lat, c_lon)
        j = int(np.argmin(d))
        if d[j] <= TOPO_NEAREST_KM:
            out.loc[i] = table.iloc[j].to_numpy()
            how.loc[i] = "nearest"
    return out, how


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(np.asarray(lon2, float) - lon1)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def load_land_mask():

    try:
        from global_land_mask import globe
        return globe
    except ImportError:
        pass
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--retries", "1",
                        "--timeout", "15", "global-land-mask"], check=True, timeout=240)
        from global_land_mask import globe
        return globe
    except Exception as e:
        print(f"[geo] global-land-mask unavailable ({type(e).__name__}: {e}) -- geo features SKIPPED")
        return None


def geo_features(df: pd.DataFrame, globe) -> pd.DataFrame:

    lat = df["Latitude"].to_numpy(float)
    lon = df["Longitude"].to_numpy(float)
    has = ~(np.isnan(lat) | np.isnan(lon))
    out = pd.DataFrame(index=df.index, dtype=float)
    cols = ["geo_is_land", "geo_coast_km"] + [f"geo_land_ring_{r}" for r in GEO_FRAC_RADII_KM]
    for c in cols:
        out[c] = np.nan
    if not has.any():
        return out

    la, lo = lat[has], lon[has]
    center = globe.is_land(la, lo).astype(bool)

    radii = np.asarray(GEO_RADII_KM, float)
    bearing = np.linspace(0, 2 * np.pi, GEO_DIRECTIONS, endpoint=False)
    dlat = radii[:, None] * np.cos(bearing)[None, :] / KM_PER_DEG
    coslat = np.clip(np.cos(np.radians(la)), 0.05, None)
    dlon = (radii[:, None] * np.sin(bearing)[None, :])[None] / (KM_PER_DEG * coslat[:, None, None])
    plat = np.clip(la[:, None, None] + dlat[None], -89.99, 89.99)
    plon = (lo[:, None, None] + dlon + 180.0) % 360.0 - 180.0
    ring = globe.is_land(plat.ravel(), plon.ravel()).reshape(plat.shape).astype(bool)

    differs = (ring != center[:, None, None]).any(axis=2)
    first = np.where(differs.any(axis=1), differs.argmax(axis=1), -1)
    dist = np.where(first >= 0, radii[np.maximum(first, 0)], GEO_FAR_KM)

    vals = {"geo_is_land": center.astype(float), "geo_coast_km": np.where(center, dist, -dist)}

    for r in GEO_FRAC_RADII_KM:
        vals[f"geo_land_ring_{r}"] = ring[:, GEO_RADII_KM.index(r), :].mean(axis=1)
    for c, v in vals.items():
        out.loc[df.index[has], c] = v
    return out


def token_features(frames: list[pd.DataFrame]) -> list[pd.DataFrame]:

    both = pd.concat(frames, ignore_index=True)
    outs = [pd.DataFrame(index=f.index) for f in frames]
    for col in MULTI_VALUED:
        split = both[col].str.split("/")
        counts = pd.Series([t.strip() for ts in split for t in ts if t.strip() and t.strip() != MISSING]).value_counts()
        vocab = sorted(counts[counts >= MIN_TOKEN_COUNT].index)
        prefix = "tok_" + re.sub(r"\W+", "_", col.lower()) + "_"
        for f, o in zip(frames, outs):
            toks = f[col].str.split("/").apply(lambda ts: {t.strip() for t in ts})
            for t in vocab:
                o[prefix + re.sub(r"\W+", "_", t.lower())] = toks.apply(lambda s, t=t: float(t in s))
    return outs


def impute_geo_from_group(train: pd.DataFrame, test: pd.DataFrame, X_tr: pd.DataFrame, X_te: pd.DataFrame,
                          cols: list[str]) -> None:

    meta = pd.concat([train[[GROUP] + [c for c, _ in GROUP_IMPUTE]], test[[GROUP] + [c for c, _ in GROUP_IMPUTE]]],
                     ignore_index=True)
    feats = pd.concat([X_tr[cols], X_te[cols]], ignore_index=True)
    located = pd.concat([X_tr["Latitude"], X_te["Latitude"]], ignore_index=True).notna().to_numpy()
    per_field = feats[located].groupby(meta.loc[located, GROUP].to_numpy()).median()
    field_meta = meta[located].drop_duplicates(GROUP).set_index(GROUP)

    for name, df, X in (("train", train, X_tr), ("test", test, X_te)):
        df["geo_source"] = np.where(X["Latitude"].notna(), "own", "none")
        for i in df.index[X["Latitude"].isna()]:
            for col, min_fields in GROUP_IMPUTE:
                v = df.at[i, col]
                if v == MISSING:
                    continue
                peers = field_meta.index[(field_meta[col] == v) & (field_meta.index != df.at[i, GROUP])]
                if len(peers) >= min_fields:
                    X.loc[i, cols] = per_field.loc[peers].median().to_numpy()
                    df.at[i, "geo_source"] = f"{col.split()[0].lower()}({len(peers)})"
                    break
        src = df["geo_source"].str.replace(r"\(\d+\)", "", regex=True).value_counts().to_dict()
        print(f"  group geo {name}: {src}")


class FeatureSet(NamedTuple):
    X_tr: pd.DataFrame
    X_te: pd.DataFrame
    base_cols: list[str]
    geo_cols: list[str]
    topo_cols: list[str]
    train: pd.DataFrame
    test: pd.DataFrame


def build_features(train: pd.DataFrame, test: pd.DataFrame, globe, ext_coords: pd.DataFrame | None,
                   topo: pd.DataFrame | None, name: str, group_impute: bool = False) -> FeatureSet:

    print(f"\n--- feature set '{name}' ---")
    train, test = clean(train), clean(test)
    impute_coords_by_field(train, test)
    if ext_coords is not None:
        apply_external_coords(train, test, ext_coords)

    tok_tr, tok_te = token_features([train, test])
    X_tr = pd.concat([train[CATEGORICAL + NUMERIC], tok_tr], axis=1)
    X_te = pd.concat([test[CATEGORICAL + NUMERIC], tok_te], axis=1)
    base_cols = list(X_tr.columns)

    geo_cols: list[str] = []
    if globe is not None:
        g_tr, g_te = geo_features(train, globe), geo_features(test, globe)
        geo_cols = list(g_tr.columns)
        X_tr = pd.concat([X_tr, g_tr], axis=1)
        X_te = pd.concat([X_te, g_te], axis=1)

    topo_cols: list[str] = []
    if topo is not None:
        t_tr, how_tr = topo_features(train, topo)
        t_te, how_te = topo_features(test, topo)
        topo_cols = list(t_tr.columns)
        X_tr = pd.concat([X_tr, t_tr], axis=1)
        X_te = pd.concat([X_te, t_te], axis=1)
        for split, df, how in (("train", train, how_tr), ("test", test, how_te)):
            has = df["Latitude"].notna().to_numpy()
            print(f"  topo join {split}: {how.value_counts().to_dict()} "
                  f"(rows with coordinates but no rings: {int((has & (how == 'none').to_numpy()).sum())})")

    if group_impute and (geo_cols or topo_cols):
        impute_geo_from_group(train, test, X_tr, X_te, geo_cols + topo_cols)
    for df, X in ((train, X_tr), (test, X_te)):
        if "geo_source" not in df.columns:
            df["geo_source"] = np.where(X["Latitude"].notna(), "own", "none")

    for split, df in (("train", train), ("test", test)):
        print(f"  coordinates {split}: {df['coord_source'].value_counts().to_dict()}")
    assert list(X_tr.columns) == list(X_te.columns)
    assert X_tr.index.equals(train.index) and X_te.index.equals(test.index)
    print(f"  features: {X_tr.shape[1]} ({len(CATEGORICAL)} categorical, {len(NUMERIC)} numeric, "
          f"{tok_tr.shape[1]} tokens, {len(geo_cols)} geo, {len(topo_cols)} topo)")
    return FeatureSet(X_tr, X_te, base_cols, geo_cols, topo_cols, train, test)
