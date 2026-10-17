import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pypinyin import lazy_pinyin

DISTRICT = {
    "朝阳": "Chaoyang", "丰台": "Fengtai", "海淀": "Haidian", "通州": "Tongzhou", "昌平": "Changping",
    "大兴": "Daxing", "顺义": "Shunyi", "房山": "Fangshan", "东城": "Dongcheng", "西城": "Xicheng",
    "石景山": "Shijingshan", "门头沟": "Mentougou", "密云": "Miyun", "亦庄开发区": "Yizhuang",
    "怀柔": "Huairou", "平谷": "Pinggu", "延庆": "Yanqing",
}
RENOVATION = {"精装": "premium", "简装": "basic", "其他": "other", "毛坯": "bare", "板楼": "slab", "板塔结合": "mixed"}
STRUCTURE = {"板楼": "slab", "塔楼": "tower", "板塔结合": "mixed", "暂无数据": "no_data", "独栋别墅": "villa", "平房": "bungalow"}
TAG = {"近地铁": "near_subway"}
DIRECTION = {"东": "E", "南": "S", "西": "W", "北": "N", "东南": "SE", "东北": "NE", "西南": "SW", "西北": "NW"}
FLOOR_LEVEL = {"低楼层": "low", "中楼层": "mid", "高楼层": "high", "顶层": "top", "底层": "ground", "地下室": "basement", "上叠": "upper_duplex"}
CJK = re.compile(r"[一-鿿]")


def strip(s):
    return s.astype("string").str.strip()


def pinyin(text):
    if pd.isna(text):
        return pd.NA
    suffix = ""
    if text.endswith("其他") and len(text) > 2:
        text, suffix = text[:-2], " Other"
    parts = []
    for tok in lazy_pinyin(text):
        tok = tok.strip()
        if tok:
            parts.append(tok[:1].upper() + tok[1:])
    return " ".join(parts) + suffix


def mapped(s, table, name):
    s = strip(s)
    unknown = sorted(set(s.dropna()) - set(table))
    assert not unknown, f"{name}: unmapped values {unknown}"
    return s.map(table).astype("string")


def layout(s):
    def one(v):
        if pd.isna(v):
            return pd.NA
        if v == "车位":
            return "parking_space"
        m = re.fullmatch(r"(\d+)(室|房间)(\d+)厅", v)
        assert m, f"layout: {v}"
        return f"{m.group(1)}{'bed' if m.group(2) == '室' else 'room'}{m.group(3)}living"
    return strip(s).map(one).astype("string")


def orientation(s):
    def one(v):
        if pd.isna(v):
            return pd.NA
        parts = v.split()
        assert all(p in DIRECTION for p in parts), f"orientation: {v}"
        return " ".join(DIRECTION[p] for p in parts)
    return strip(s).map(one).astype("string")


def floor(s):
    def one(v):
        if pd.isna(v):
            return pd.NA
        m = re.fullmatch(r"(\S+?)\(共(\d+)层\)", v)
        if m:
            assert m.group(1) in FLOOR_LEVEL, f"floor: {v}"
            return f"{FLOOR_LEVEL[m.group(1)]}({m.group(2)}F total)"
        m = re.fullmatch(r"(\d+)层", v)
        assert m, f"floor: {v}"
        return f"F{m.group(1)}"
    return strip(s).map(one).astype("string")


def number(s, pattern):
    def one(v):
        if pd.isna(v):
            return np.nan
        m = re.fullmatch(pattern, str(v).strip().replace(",", ""))
        assert m, f"{pattern}: {v}"
        return float(m.group(1))
    return s.map(one)


def translate(z):
    out = pd.DataFrame({
        "row_id": np.arange(len(z)),
        "community": z["小区名"].astype("string").str.strip().map(pinyin).astype("string"),
        "community_zh": strip(z["小区名"]),
        "street": z["所在街道或镇"].astype("string").str.strip().map(pinyin).astype("string"),
        "street_zh": strip(z["所在街道或镇"]),
        "district": mapped(z["所在区"], DISTRICT, "district"),
        "layout": layout(z["户型"]),
        "area_sqm": number(z["面积"], r"([\d.]+)平米"),
        "orientation": orientation(z["朝向"]),
        "renovation": mapped(z["装修"], RENOVATION, "renovation"),
        "floor": floor(z["楼层"]),
        "year_built": number(z["年份"], r"(\d{4})年建").astype("Int64"),
        "structure": mapped(z["结构"], STRUCTURE, "structure"),
        "tag": mapped(z["房源标签"], TAG, "tag"),
    })
    if "单价" in z:
        out["total_price_10k_cny"] = number(z["总价"], r"([\d.]+)万")
        out["unit_price"] = number(z["单价"], r"(\d+)元/平").astype("int64")
    return out


def align(full, pre, area_col):
    keys = list(zip(full["community_zh"], full["area_sqm"].round(2)))
    pre_keys = list(zip(pre["community_name"].astype(str).str.strip(), pre[area_col].astype(float).round(2)))
    idx, j = [], 0
    for k in pre_keys:
        while keys[j] != k:
            j += 1
        idx.append(j)
        j += 1
    return np.array(idx)


def main(raw, out):
    raw, out = Path(raw), Path(out)
    z_tr = pd.read_excel(raw / "archive" / "_.xlsx")
    z_te = pd.read_excel(raw / "archive" / "__.xlsx")
    train, test = translate(z_tr), translate(z_te)

    e_tr = pd.read_excel(raw / "housing_train_set.xlsx")
    e_te = pd.read_excel(raw / "housing_test_set.xlsx")
    assert (e_tr["community_name"].astype(str).str.strip().values == train["community_zh"].values).all()
    assert (e_te["community_name"].astype(str).str.strip().values == test["community_zh"].values).all()
    assert (e_tr["unit_price"].str.replace(" CNY/sqm", "").astype(int).values == train["unit_price"].values).all()

    p_tr = pd.read_excel(raw / "housing_train_preprocessed.xlsx")
    p_te = pd.read_excel(raw / "housing_test_preprocessed.xlsx")
    tr_idx, te_idx = align(train, p_tr, "area"), align(test, p_te, "area")
    assert (train["unit_price"].values[tr_idx] == p_tr["unit_price"].values).all()
    assert len(te_idx) == 581
    train.insert(1, "organiser_kept", 0)
    train.loc[tr_idx, "organiser_kept"] = 1
    test.insert(1, "ID", pd.array([pd.NA] * len(test), dtype="Int64"))
    test.loc[te_idx, "ID"] = np.arange(len(te_idx))

    for name, df in [("train", train), ("test", test)]:
        for c in df.columns:
            if c.endswith("_zh"):
                continue
            assert not df[c].astype(str).str.contains(CJK).any(), f"{name}.{c} still has Chinese"
        df.to_csv(out / f"{name}.csv", index=False)
    sub = pd.DataFrame({"ID": np.arange(581), "单价": int(round(train["unit_price"].mean()))})
    sub.to_csv(out / "sample_submission.csv", index=False)
    print(train.shape, test.shape, "organiser kept", int(train["organiser_kept"].sum()), int(test["ID"].notna().sum()))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
