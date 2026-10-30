import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

SNAPSHOT = pd.Timestamp("2026-09-19")
N_AMENITIES = 150
MIN_LOCATION = 30
FLAGS = ["host_is_superhost", "host_has_profile_pic", "host_identity_verified"]
CATS = ["property_type", "room_type", "host_location"]
TE_KEYS = [["geo_cell"], ["host_id"], ["geo_cell", "room_type"], ["host_id", "description"], ["host_id", "name"]]
NAME_WORDS = {"studio": r"\bstudio\b", "penthouse": r"\bpenthouse\b", "luxury": r"\bluxur", "hostel": r"\bhostel|\bdorm",
              "double_room": r"\bdouble room\b", "single": r"\bsingle (?:room|bed)\b", "ensuite": r"\ben.?suite\b",
              "garden": r"\bgarden\b", "terrace": r"\bterrace|\bbalcon", "view": r"\bview", "cosy": r"\bco[sz]y\b",
              "family": r"\bfamily\b", "townhouse": r"\btown ?house\b", "cabin": r"\bcabin|\bpod\b"}


def find_data():
    root = Path("/kaggle/input")
    found = {}
    for p in sorted(root.rglob("*.csv")) if root.exists() else []:
        cols = set(pd.read_csv(p, nrows=0).columns)
        if {"id", "accommodates", "geo_cell"} <= cols:
            found["train" if "price" in cols else "test"] = p
        elif {"geo_cell", "geo_x", "geo_y"} <= cols:
            found["geo"] = p
    if len(found) == 3:
        return found["train"], found["test"], found["geo"]
    try:
        d = Path(__file__).resolve().parent.parent / "data"
    except NameError:
        d = Path.cwd() / "data"
    return d / "train.csv", d / "test.csv", d / "ext_geo.csv"


def load():
    p_tr, p_te, p_geo = find_data()
    tr, te, geo = pd.read_csv(p_tr), pd.read_csv(p_te), pd.read_csv(p_geo)
    print(f"train {p_tr} {tr.shape}, test {p_te} {te.shape}, geo {p_geo} {geo.shape}")
    return tr, te, geo


def baths(text):
    s = text.fillna("").str.lower()
    num = s.str.extract(r"(\d+(?:\.\d+)?)")[0].astype(float)
    num = num.where(~s.str.contains("half-bath"), 0.5)
    num = num.where(text.notna())
    return num, s.str.contains("shared").astype(int), s.str.contains("private").astype(int)


def clean(s):
    return s.fillna("").str.replace(r"<[^>]+>", " ", regex=True)


def base_block(df):
    X = df.select_dtypes("number").drop(columns=["id", "host_id", "geo_cell"]).copy()
    X["host_user_months"] = df.hosts_time_as_user_years * 12 + df.hosts_time_as_user_months
    X["host_host_months"] = df.hosts_time_as_host_years * 12 + df.hosts_time_as_host_months
    for c in FLAGS:
        X[c] = (df[c] == "t").astype(int)
    X["baths"], X["bath_shared"], X["bath_private"] = baths(df.bathrooms_text)
    for c in ["first_review", "last_review"]:
        X[f"days_{c}"] = (SNAPSHOT - pd.to_datetime(df[c])).dt.days
    X["people_per_bedroom"] = df.accommodates / df.bedrooms.replace(0, np.nan)
    X["baths_per_bedroom"] = X.baths / df.bedrooms.replace(0, np.nan)
    X["private_room_share"] = df.calculated_host_listings_count_private_rooms / df.calculated_host_listings_count

    am = df.amenities.map(json.loads)
    X["n_amenities"] = am.map(len)
    top = [a for a, _ in Counter(a for row in am for a in set(row)).most_common(N_AMENITIES)]
    sets = am.map(set)
    X = pd.concat([X, pd.DataFrame({f"am_{i}": sets.map(lambda s, a=a: a in s).astype(np.int8)
                                    for i, a in enumerate(top)})], axis=1)

    desc = df.description.fillna("")
    X["name_len"], X["desc_len"] = df.name.str.len(), desc.str.len()
    X["desc_words"] = desc.str.split().map(len)
    X["name_upper"] = df.name.map(lambda s: sum(ch.isupper() for ch in s) / max(len(s), 1))
    X["sqm"] = desc.str.extract(r"(\d{2,4})\s*(?:sq\.?\s*m|m2|m²|square met)", flags=re.I)[0].astype(float)
    X["sqft"] = desc.str.extract(r"(\d{3,5})\s*(?:sq\.?\s*f|square f)", flags=re.I)[0].astype(float)

    for c in ["geo_cell", "host_id"]:
        X[f"cnt_{c}"] = df.groupby(c)[c].transform("size")
    X["cnt_host_desc"] = df.groupby(["host_id", "description"], dropna=False).id.transform("size")

    loc = df.host_location.where(df.host_location.map(df.host_location.value_counts()) >= MIN_LOCATION)
    for c, s in zip(CATS, [df.property_type, df.room_type, loc]):
        X[c] = s.astype("category")
    return X


def mined_block(df):
    desc, name = clean(df.description).str.lower(), df.name.str.lower()
    X = pd.DataFrame(index=df.index)
    X["host_missing"] = df.calculated_host_listings_count - df.groupby("host_id").id.transform("size")
    X["host_missing_share"] = X.host_missing / df.calculated_host_listings_count
    X["floor"] = desc.str.extract(r"(\d{1,2})(?:st|nd|rd|th)\s+floor")[0].astype(float)
    X["walk_min"] = desc.str.extract(r"(\d{1,2})\s*(?:min|minute)s?\s*walk")[0].astype(float)
    X["sleeps"] = desc.str.extract(r"sleeps\s+(?:up to\s+)?(\d{1,2})")[0].astype(float)
    X["name_beds"] = name.str.extract(r"\b(\d)\s*-?\s*(?:bed|br\b|bedroom)")[0].astype(float)
    X["name_beds_gap"] = X.name_beds - df.bedrooms
    X["fee_mention"] = desc.str.contains(r"£\s*\d").astype(int)
    for k, pat in NAME_WORDS.items():
        X[f"w_{k}"] = name.str.contains(pat).astype(int)
    return X


def geo_block(df, geo):
    g = df[["geo_cell"]].merge(geo, on="geo_cell", how="left").drop(columns="geo_cell")
    g["geo_bearing"] = np.arctan2(g.geo_y, g.geo_x)
    return g


def features(tr, te, geo):
    df = pd.concat([tr.drop(columns="price"), te], ignore_index=True)
    blocks = {"base": base_block(df), "mined": mined_block(df), "geo": geo_block(df, geo)}
    for X in blocks.values():
        X.columns = [re.sub(r"[^0-9a-zA-Z_]", "_", c) for c in X.columns]
    keys = df[sorted({k for key in TE_KEYS for k in key})].copy()
    text = pd.DataFrame({"name": df.name, "desc": clean(df.description),
                         "amen": df.amenities.map(lambda s: "\t".join(json.loads(s))),
                         "cats": "pt_" + df.property_type.str.replace(" ", "_") + " rt_" + df.room_type.str.replace(" ", "_")
                                 + " gc_" + df.geo_cell.astype(str) + " h_" + df.host_id.astype(str)})
    cells = geo.set_index("geo_cell")[["geo_x", "geo_y"]]
    print("blocks: " + ", ".join(f"{k} {v.shape[1]}" for k, v in blocks.items())
          + f"; geo coverage {blocks['geo'].geo_x.notna().mean():.3f}")
    return blocks, keys, text, cells, np.log1p(tr.price.values), te.id.values
