from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.neighbors import BallTree

REF_YEAR = 2024
LEVELS = {"basement": -1, "ground": 0, "low": 1, "mid": 2, "high": 3, "top": 4, "upper_duplex": 4}
SOUTH = {"S", "SE", "SW"}
NORTH = {"N", "NE", "NW"}
NUM_COLS = [
    "bedrooms", "living_rooms", "is_room_layout", "is_parking", "area", "log_area", "area_per_room",
    "floor_total", "floor_level_ord", "age", "faces_south", "faces_north", "through", "faces_s_exact",
    "n_directions", "near_subway", "name_villa", "name_apartment", "name_old", "name_phase", "name_subarea",
]
NAME_PATTERNS = {
    "name_villa": r"墅", "name_apartment": r"公寓", "name_old": r"里$|号院|胡同|宿舍|家属",
    "name_phase": r"期", "name_subarea": r"[东西南北一二三四五六七八九十\d]区$",
}
EXT_LOG_COLS = [
    "px_combined", "px_ariosolzq", "px_zhangzhenzhe", "px_xiaoyusmd", "px_msq0313", "px_chrom826",
    "rent_yokumii", "rent_caleyuexing", "txn_px_k30", "txn_px_300m", "spx_combined", "srent_yokumii",
    "srent_caleyuexing", "dist_center_km", "dist_subway_m", "txn_dist_k30_m", "px_onceonceonce", "px_comp",
    "dist_school_m", "dist_top_school_m", "dist_university_m", "dist_hospital_m", "dist_park_m", "dist_mall_m",
    "px_txn_comp_200m", "px_txn_comp_500m",
]
EXT_RAW_COLS = [
    "px_n_sources", "spx_n_sources", "lat", "lng", "coord_level", "n_subway_1km", "n_subway_2km", "txn_n_300m",
    "n_comp", "ring_zone", "n_school_1km", "n_kindergarten_1km", "n_office_1km", "n_supermarket_1km",
    "n_university_2km", "n_hospital_2km", "n_park_2km", "n_top_school_2km", "n_txn_comp_200m", "n_txn_comp_500m",
]
GROUP_COLS = [
    "comm_n_listings", "comm_mean_area", "comm_max_floors", "comm_mean_age", "comm_room_share",
    "comm_subway_share", "area_rel", "age_rel", "street_n_listings",
]
EXT_COLS = [f"log_{c}" for c in EXT_LOG_COLS] + EXT_RAW_COLS
LOW_CARD_COLS = ["district", "floor_level", "renovation", "structure"]
CAT_COLS = LOW_CARD_COLS + ["street_key", "community_key"]
HIER_LEVELS = ["district", "street_key", "community_key"]
HIER_TE_COLS = ["te_district", "te_street", "te_community", "n_street", "n_community"]
SPATIAL_COLS = ["sp_knn_price", "sp_knn_dist_km", "sp_knn_nearest_km"]
TE_COLS = HIER_TE_COLS + SPATIAL_COLS
V2_NUM_COLS = [
    "bedrooms", "living_rooms", "is_room_layout", "is_parking", "area", "log_area", "area_per_room",
    "floor_total", "floor_level_ord", "age", "faces_south", "faces_north", "through", "faces_s_exact",
    "n_directions", "near_subway",
]
V2_EXT_COLS = [f"log_{c}" for c in [
    "px_combined", "px_ariosolzq", "px_zhangzhenzhe", "px_xiaoyusmd", "px_msq0313", "px_chrom826",
    "rent_yokumii", "rent_caleyuexing", "txn_px_k30", "txn_px_300m", "spx_combined", "srent_yokumii",
    "srent_caleyuexing", "dist_center_km", "dist_subway_m", "txn_dist_k30_m",
]] + ["px_n_sources", "spx_n_sources", "lat", "lng", "coord_level", "n_subway_1km", "n_subway_2km", "txn_n_300m"]
POOL_CAT = ["source", "district", "pool_biz", "pool_comm", "floor_level", "renovation", "structure"]
POOL_LOC_CAT = ["source", "district", "floor_level", "renovation", "structure"]
POOL_NUM = [
    "area", "log_area", "bedrooms", "living_rooms", "is_room_layout", "floor_total", "floor_level_ord", "year_built",
    "faces_south", "faces_north", "through", "n_directions", "near_subway", "lat", "lng", "coord_level",
    "dist_center_km", "ring_zone", "dist_subway_m", "n_subway_1km",
]
POOL_REF_SOURCES = ["onceonceonce", "msq0313", "xiaoyusmd", "ariosolzq"]
POOL_FEATURES = ["log_pool_px", "log_pool_px_loc", "pool_comm_premium"]
SPATIAL_K = 8


def get_data_dir():
    kaggle = Path("/kaggle/input")
    if kaggle.exists():
        for p in sorted(kaggle.rglob("train.csv")):
            with open(p, encoding="utf-8") as f:
                if "community_zh" in f.readline():
                    return p.parent
        raise FileNotFoundError("no train.csv with a community_zh column under /kaggle/input")
    try:
        return Path(__file__).resolve().parent.parent / "data"
    except NameError:
        return Path.cwd() / "data"


def get_out_dir():
    return Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")


def load_data():
    d = get_data_dir()
    print(f"data dir: {d}")
    train, test = pd.read_csv(d / "train.csv"), pd.read_csv(d / "test.csv")
    comm, street = pd.read_csv(d / "ext_community.csv"), pd.read_csv(d / "ext_street.csv")
    comp = pd.read_csv(d / "ext_comparable.csv")
    assert comm["community_zh"].is_unique and street["street_zh"].is_unique
    assert not comp.duplicated(["community_zh", "area_sqm"]).any()
    out = []
    for df in (train, test):
        n = len(df)
        df = df.merge(comm, on="community_zh", how="left").merge(street, on="street_zh", how="left")
        df = df.merge(comp, on=["community_zh", "area_sqm"], how="left")
        assert len(df) == n
        out.append(df)
    print(f"external: px_combined coverage train {out[0]['px_combined'].notna().mean():.3f}, "
          f"test {out[1]['px_combined'].notna().mean():.3f}")
    return out[0], out[1]


def load_pool():
    pool = pd.read_csv(get_data_dir() / "ext_pool.csv", low_memory=False)
    assert pool.loc[pool["source"] == "competition", "price"].isna().all()
    pool["log_area"] = np.log(pool["area"])
    pool["floor_level_ord"] = pool["floor_level"].map(LEVELS)
    for c in POOL_CAT:
        pool[c] = pool[c].fillna("missing").astype(str)
    print(f"pool: {int((pool['source'] != 'competition').sum())} external rows, "
          f"{int((pool['source'] == 'competition').sum())} competition rows")
    return pool


def add_pool_features(F, df, pool_pred, split):
    keys = split + ":" + df["row_id"].astype(str)
    for c in ["log_pool_px", "log_pool_px_loc"]:
        F[c] = keys.map(pool_pred.set_index("row_key")[c]).to_numpy()
    F["pool_comm_premium"] = F["log_pool_px"] - F["log_pool_px_loc"]
    return F


def build_features(df):
    out = pd.DataFrame(index=df.index)
    layout = df["layout"].fillna("")
    lay = layout.str.extract(r"^(\d+)(bed|room)(\d+)living$")
    out["bedrooms"] = pd.to_numeric(lay[0])
    out["living_rooms"] = pd.to_numeric(lay[2])
    out["is_room_layout"] = (lay[1] == "room").astype(int)
    out["is_parking"] = (layout == "parking_space").astype(int)
    out["area"] = df["area_sqm"].astype(float)
    out["log_area"] = np.log(out["area"])
    out["area_per_room"] = out["area"] / (out["bedrooms"] + out["living_rooms"]).clip(lower=1)

    floor = df["floor"].fillna("")
    fl = floor.str.extract(r"^([a-z_]+)\((\d+)F total\)$")
    fx = floor.str.extract(r"^F(\d+)$")[0]
    level = fl[0].copy()
    level[fx.notna()] = "unstated"
    level = level.fillna("missing")
    out["floor_total"] = pd.to_numeric(fl[1]).fillna(pd.to_numeric(fx))
    out["floor_level_ord"] = level.map(LEVELS)

    year = df["year_built"].where(df["year_built"] <= REF_YEAR)
    out["age"] = REF_YEAR - year

    tokens = df["orientation"].fillna("").str.split()
    south = tokens.map(lambda t: any(x in SOUTH for x in t))
    north = tokens.map(lambda t: any(x in NORTH for x in t))
    out["faces_south"] = south.astype(int)
    out["faces_north"] = north.astype(int)
    out["through"] = (south & north).astype(int)
    out["faces_s_exact"] = tokens.map(lambda t: "S" in t).astype(int)
    out["n_directions"] = tokens.map(len)
    out["near_subway"] = df["tag"].eq("near_subway").astype(int)
    for c, pattern in NAME_PATTERNS.items():
        out[c] = df["community_zh"].fillna("").str.contains(pattern).astype(int)

    for c in EXT_LOG_COLS:
        out[f"log_{c}"] = np.log1p(df[c].astype(float))
    for c in EXT_RAW_COLS:
        out[c] = df[c].astype(float)

    out["district"] = df["district"].fillna("missing")
    out["floor_level"] = level
    out["renovation"] = df["renovation"].where(~df["renovation"].isin(["slab", "mixed"])).fillna("missing")
    out["structure"] = df["structure"].fillna("missing")
    out["street_key"] = out["district"] + "|" + df["street_zh"].fillna("missing")
    out["community_key"] = out["district"] + "|" + df["community_zh"].fillna("missing")
    return out


def add_group_features(Ftr, Fte):
    both = pd.concat([Ftr, Fte], keys=["tr", "te"])
    g = both.groupby("community_key")
    both["comm_n_listings"] = g["area"].transform("size")
    both["comm_mean_area"] = g["area"].transform("mean")
    both["comm_max_floors"] = g["floor_total"].transform("max")
    both["comm_mean_age"] = g["age"].transform("mean")
    both["comm_room_share"] = g["is_room_layout"].transform("mean")
    both["comm_subway_share"] = g["near_subway"].transform("mean")
    both["area_rel"] = both["area"] / both["comm_mean_area"]
    both["age_rel"] = both["age"] - both["comm_mean_age"]
    both["street_n_listings"] = both.groupby("street_key")["area"].transform("size")
    return both.loc["tr"], both.loc["te"]


class SpatialNeighbourEncoder:
    def __init__(self, k=SPATIAL_K):
        self.k = k

    def fit(self, X, y):
        d = pd.DataFrame({"key": X["community_key"].to_numpy(), "lat": X["lat"].to_numpy(),
                          "lng": X["lng"].to_numpy(), "y": np.asarray(y, dtype=float)})
        d = d.dropna(subset=["lat", "lng"])
        self.comm_ = d.groupby("key").agg(lat=("lat", "first"), lng=("lng", "first"), y=("y", "mean"))
        self.tree_ = BallTree(np.radians(self.comm_[["lat", "lng"]].to_numpy()), metric="haversine")
        return self

    def transform(self, X):
        out = pd.DataFrame(np.nan, index=X.index, columns=SPATIAL_COLS)
        ok = X["lat"].notna().to_numpy() & X["lng"].notna().to_numpy()
        if not ok.any():
            return out
        k = min(self.k + 1, len(self.comm_))
        dist, idx = self.tree_.query(np.radians(X.loc[ok, ["lat", "lng"]].to_numpy()), k=k)
        dist = dist * 6371.0
        keys = self.comm_.index.to_numpy()[idx]
        own = keys == X.loc[ok, "community_key"].to_numpy()[:, None]
        rank = np.cumsum(~own, axis=1)
        use = ~own & (rank <= self.k)
        yv = self.comm_["y"].to_numpy()[idx]
        w = np.where(use, 1.0 / (dist + 0.2), 0.0)
        out.loc[ok, "sp_knn_price"] = (w * yv).sum(axis=1) / w.sum(axis=1)
        out.loc[ok, "sp_knn_dist_km"] = np.where(use, dist, 0).sum(axis=1) / use.sum(axis=1)
        out.loc[ok, "sp_knn_nearest_km"] = np.where(use, dist, np.inf).min(axis=1)
        return out


class HierTargetEncoder:
    def __init__(self, levels=HIER_LEVELS):
        self.levels = levels

    @staticmethod
    def _eb_k(key, y, prior):
        g = pd.DataFrame({"key": key, "y": y, "prior": prior})
        agg = g.groupby("key").agg(n=("y", "size"), mean=("y", "mean"), prior=("prior", "mean"))
        dev2 = (g["y"] - g.groupby("key")["y"].transform("mean")) ** 2
        dof = (agg["n"] - 1).sum()
        s_w2 = dev2.sum() / dof if dof > 0 else np.var(y)
        tau2 = ((agg["mean"] - agg["prior"]) ** 2 - s_w2 / agg["n"]).mean()
        tau2 = max(tau2, 1e-6 * np.var(y))
        return float(np.clip(s_w2 / tau2, 0.1, 100.0))

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        self.global_ = y.mean()
        self.stats_, self.k_ = [], []
        prior = np.full(len(y), self.global_)
        for col in self.levels:
            key = X[col].to_numpy()
            k = self._eb_k(key, y, prior)
            stats = pd.DataFrame({"key": key, "y": y}).groupby("key")["y"].agg(["size", "sum"])
            n = stats["size"].reindex(key).to_numpy(dtype=float)
            s = stats["sum"].reindex(key).to_numpy(dtype=float)
            prior = (s + k * prior) / (n + k)
            self.stats_.append(stats)
            self.k_.append(k)
        return self

    def transform(self, X):
        prior = np.full(len(X), self.global_)
        out = {}
        names = {"district": "district", "street_key": "street", "community_key": "community"}
        for col, stats, k in zip(self.levels, self.stats_, self.k_):
            key = X[col].to_numpy()
            n = stats["size"].reindex(key).fillna(0).to_numpy(dtype=float)
            s = stats["sum"].reindex(key).fillna(0).to_numpy(dtype=float)
            prior = (s + k * prior) / (n + k)
            out[f"te_{names[col]}"] = prior
            out[f"n_{names[col]}"] = n
        df = pd.DataFrame(out, index=X.index)
        return df[HIER_TE_COLS]


def encode_train_test(Xtr, ytr, Xte, n_inner, seed):
    ytr = np.asarray(ytr, dtype=float)
    te_tr = pd.DataFrame(np.nan, index=Xtr.index, columns=TE_COLS)
    for a, b in KFold(n_inner, shuffle=True, random_state=seed).split(Xtr):
        enc = HierTargetEncoder().fit(Xtr.iloc[a], ytr[a])
        sp = SpatialNeighbourEncoder().fit(Xtr.iloc[a], ytr[a])
        te_tr.iloc[b] = pd.concat([enc.transform(Xtr.iloc[b]), sp.transform(Xtr.iloc[b])], axis=1).to_numpy()
    enc = HierTargetEncoder().fit(Xtr, ytr)
    sp = SpatialNeighbourEncoder().fit(Xtr, ytr)
    return te_tr, pd.concat([enc.transform(Xte), sp.transform(Xte)], axis=1), enc
