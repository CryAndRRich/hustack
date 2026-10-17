import json
import re
import sys
from pathlib import Path

import lxml.html
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from preprocess import EDU_COLS, NO_EDU, city_regex, norm_edu

RAW = ROOT / "data" / "external" / "raw"
OUT = ROOT / "data" / "external"
FUZZ_MIN = {"wd": 90, "hse": 90, "raex": 90}
ACRONYM_PREFIX = r"^(ниу|нияу|рту|ни|ru|фгбоу во|гоу впо)\s+"
GENERIC = {"государственный", "университет", "институт", "национальный", "исследовательский", "имени", "им",
           "федеральный", "академия", "университета", "института", "высшего", "образования", "профессионального",
           "учреждение", "бюджетное", "автономное", "федеральное", "образовательное", "филиал"}
MANUAL_ALIAS = {
    "санкт петербургский национальный исследовательский университет информационных технологий механики и оптики":
        "университет итмо",
    "санкт петербургский государственный университет информационных технологий механики и оптики":
        "университет итмо",
    "санкт петербургский государственный горный университет": "санкт петербургский горный университет",
}
SAFE_SHORT_ACRONYMS = {"мгу", "рэу", "вшэ"}
REGION_ALIAS = {"удмуртия": "удмуртская", "чечня": "чеченская", "чувашия": "чувашская"}
REGION_STOP = {"республика", "респ", "область", "обл", "край", "автономный", "автономная", "округ", "ао",
               "г", "город", "федерального", "значения", "югра", "кузбасс", "алания", "якутия"}


def region_key(s):
    toks = [t.strip(".,*") for t in re.split(r"[\s\-—–/()]+", str(s).lower().replace("ё", "е"))]
    toks = [t for t in toks if t and t not in REGION_STOP]
    return REGION_ALIAS.get(toks[0], toks[0]) if toks else ""


def num(s):
    s = re.sub(r"&+[\d.]+", "", str(s))
    s = s.replace("\xa0", "").replace(" ", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return np.nan


def html_rows(path, table_index=None):
    d = lxml.html.fromstring(path.read_bytes().decode("utf-8", "ignore"))
    tables = d.findall(".//table")
    t = tables[table_index] if table_index is not None else max(tables, key=lambda x: len(x.findall(".//tr")))
    return [[c.text_content().strip() for c in r if c.tag in ("td", "th")] for r in t.findall(".//tr")]


def build_cities():
    c = pd.read_csv(RAW / "hflabs_city.csv")
    c["city"] = c["city"].fillna(c["settlement"]).fillna(c["area"]).fillna(c["region"])
    c = c[["city", "region", "region_type", "federal_district", "population", "geo_lat", "geo_lon"]].copy()
    c["region_key"] = c["region"].map(region_key)

    rows = html_rows(RAW / "wiki_income.html", 1)
    wage = {region_key(r[0]): num(r[1]) for r in rows[1:] if len(r) >= 2 and not np.isnan(num(r[1]))}
    rows = html_rows(RAW / "wiki_grp_pc.html", 0)
    grp = {region_key(r[1]): num(r[-1]) for r in rows[1:] if len(r) >= 3}
    c["region_wage"] = c["region_key"].map(wage)
    c["region_grp_pc"] = c["region_key"].map(grp)
    for fname, col in [("rosstat_wage_2023_02.xlsx", "region_teacher_wage"),
                       ("rosstat_wage_2023_03.xlsx", "region_addl_teacher_wage"),
                       ("rosstat_wage_2023_05.xlsx", "region_univ_wage")]:
        x = pd.read_excel(RAW / fname, header=None)
        x = x[x[0].notna() & pd.to_numeric(x[1], errors="coerce").notna()]
        keys = x[0].astype(str).str.replace(r"^\s*г\.\s*", "", regex=True).map(region_key)
        vals = dict(zip(keys[::-1], pd.to_numeric(x[1], errors="coerce")[::-1]))
        c[col] = c["region_key"].map(vals)
        if col == "region_teacher_wage":
            all_wage = dict(zip(keys[::-1], pd.to_numeric(x[x.columns[-1]], errors="coerce")[::-1]))
            c["region_wage_2023"] = c["region_key"].map(all_wage)
    for col in ["region_teacher_wage", "region_addl_teacher_wage", "region_univ_wage", "region_wage_2023"]:
        print(f"        regions without {col}: {sorted(set(c.loc[c[col].isna(), 'region']))}")
    miss = sorted(set(c.loc[c["region_wage"].isna(), "region"]))
    print(f"cities: {len(c)}; regions without wage: {miss}")
    miss = sorted(set(c.loc[c["region_grp_pc"].isna(), "region"]))
    print(f"        regions without GRP: {miss}")
    c = c.sort_values("population", ascending=False).drop_duplicates("city")
    c.drop(columns=["region_key", "region_type"]).to_csv(OUT / "ext_cities.csv", index=False)
    return c


def build_metro():
    rows = []
    for city in json.loads((RAW / "hh_metro.json").read_text()):
        name = re.sub(r"\s*\(.*\)", "", city["name"])
        for line in city["lines"]:
            for st in line["stations"]:
                rows.append((st["name"], name))
    m = pd.DataFrame(rows, columns=["station", "city"]).drop_duplicates()
    m.to_csv(OUT / "ext_metro.csv", index=False)
    print(f"metro: {len(m)} stations in {m.city.nunique()} cities")


def nearest_city(lat, lon, cities):
    if np.isnan(lat):
        return None, np.nan
    la, lo = np.radians(cities["geo_lat"].to_numpy()), np.radians(cities["geo_lon"].to_numpy())
    a = np.sin((la - np.radians(lat)) / 2) ** 2 + np.cos(la) * np.cos(np.radians(lat)) * np.sin((lo - np.radians(lon)) / 2) ** 2
    d = 2 * 6371 * np.arcsin(np.sqrt(a)) - 0.19 * np.sqrt(cities["population"].to_numpy() / 1000)
    i = int(np.argmin(d))
    return cities["city"].iloc[i], d[i]


def parse_point(s):
    m = re.match(r"Point\(([-\d.]+) ([-\d.]+)\)", str(s))
    return (float(m.group(2)), float(m.group(1))) if m else (np.nan, np.nan)


def year(s):
    m = re.match(r"(-?\d{4})", str(s))
    return float(m.group(1)) if m else np.nan


def wikidata_table(cities):
    ru = pd.read_csv(RAW / "wd_univ_ru.csv")
    ru["country"] = "Q159"
    cis = pd.read_csv(RAW / "wd_univ_cis.csv")
    cis["country"] = cis["country"].str.rsplit("/", n=1).str[-1]
    wd = pd.concat([ru, cis], ignore_index=True)
    wd["id"] = wd["u"].str.rsplit("/", n=1).str[-1]
    wd[["lat", "lon"]] = wd["coord"].map(parse_point).apply(pd.Series)
    wd["inception"] = wd["inception"].map(year)
    agg = wd.groupby("id").agg(label=("uLabel", "first"), country=("country", "first"), lat=("lat", "first"),
                               p131=("cityLabel", "first"),
                               lon=("lon", "first"), inception=("inception", "min"), students=("students", "max"))
    ru_cities = cities[cities["population"] >= 20000]
    near = [nearest_city(a, b, ru_cities) for a, b in zip(agg["lat"], agg["lon"])]
    known = set(cities["city"])
    agg["city"] = [p if isinstance(p, str) and p in known else (c if d <= 25 else None)
                   for p, (c, d) in zip(agg["p131"], near)]
    al = pd.read_csv(RAW / "wd_univ_alias.csv")
    al["id"] = al["u"].str.rsplit("/", n=1).str[-1]
    names = pd.concat([
        agg["label"].rename("name").reset_index(),
        al[["id", "alias"]].rename(columns={"alias": "name"}),
        al[["id", "short"]].rename(columns={"short": "name"}),
    ]).dropna()
    names["key"] = names["name"].map(norm_edu)
    names = names[names["key"].str.len() >= 3].drop_duplicates(["key", "id"])
    names = names.merge(agg[["students"]], left_on="id", right_index=True)
    names = names.sort_values("students", ascending=False).drop_duplicates("key")
    print(f"wikidata: {len(agg)} institutions, {len(names)} name keys, {agg['city'].notna().sum()} placed in a Russian city")
    return agg, names[["key", "id"]]


def hse_table():
    frames = []
    for fid, kind in [("98607722", "budget"), ("98607745", "paid")]:
        rows = html_rows(RAW / f"hse_rating_2024_{fid}.html")
        df = pd.DataFrame([r[:2] for r in rows[1:] if len(r) >= 2], columns=["name", f"ege_{kind}"])
        df[f"ege_{kind}"] = df[f"ege_{kind}"].map(num)
        frames.append(df.set_index("name"))
    h = frames[0].join(frames[1], how="outer").reset_index()
    h["city"] = h["name"].str.extract(r"г\.\s*([А-ЯЁ][А-Яа-яё\-]+(?:[\s\-]на[\s\-][А-ЯЁ][а-яё]+)?)\s*$")[0]
    h["key"] = h["name"].str.replace(r",?\s*г\.\s*[^,]+$", "", regex=True).map(norm_edu)
    h["filial"] = h["name"].str.lower().str.contains("филиал").astype(int)
    h = h.drop_duplicates("key")
    print(f"hse: {len(h)} universities with 2024 ЕГЭ admission scores")
    return h


def raex_table():
    out = {}
    for y in (2024, 2025, 2026):
        for r in html_rows(RAW / f"raex100_{y}.html")[1:]:
            if len(r) >= 2 and r[0].isdigit():
                k = norm_edu(r[1])
                out.setdefault(k, []).append(int(r[0]))
    t = pd.DataFrame([(k, float(np.mean(v)), len(v)) for k, v in out.items()], columns=["key", "raex_rank", "raex_years"])
    print(f"raex: {len(t)} universities ranked in 2024-2026")
    return t


def distinct(k):
    return {t for t in k.split() if len(t) >= 3 and t not in GENERIC}


ADJ_END = re.compile(r"(ский|цкий|ской|ный|ной|ний|ий|ый|ая)$")


def compatible(a, b):
    da, db = distinct(a), distinct(b)
    if not da or not db or not (da <= db or db <= da):
        return False
    return not any(ADJ_END.search(t) for t in da ^ db)


def eponym(k):
    m = re.search(r"\b(?:им|имени)\b(.*)$", k)
    toks = [t for t in m.group(1).split() if len(t) >= 4] if m else []
    return toks[-1] if toks else None


def eponym_clash(k, names):
    e = eponym(k)
    others = {eponym(n) for n in names} - {None}
    return e is not None and bool(others) and e not in others


def match(keys, cand_keys, cutoff):
    exact = set(cand_keys)
    res = {}
    for k in keys:
        k2 = re.sub(ACRONYM_PREFIX, "", k)
        k3 = MANUAL_ALIAS.get(k, "")
        head = k2.split()[0] if k2 else ""
        if k in exact:
            res[k] = (k, 100.0)
        elif k2 in exact:
            res[k] = (k2, 100.0)
        elif k3 in exact:
            res[k] = (k3, 99.0)
        elif (len(head) >= 4 or head in SAFE_SHORT_ACRONYMS) and head not in GENERIC and not ADJ_END.search(head) \
                and head in exact:
            res[k] = (head, 98.0)
        else:
            hits = process.extract(k, cand_keys, scorer=fuzz.token_sort_ratio, score_cutoff=cutoff, limit=5)
            hits = [h for h in hits if compatible(k, h[0])]
            res[k] = (hits[0][0], float(hits[0][1])) if hits else (None, 0.0)
    return res


def adjective_city_map(wd, names):
    lab = wd.dropna(subset=["city"]).copy()
    lab["adj"] = lab["label"].map(norm_edu).str.split().str[0]
    lab = lab[lab["adj"].str.endswith(("ский", "цкий", "ской"))]
    out = {}
    for adj, g in lab.groupby("adj"):
        vc = g["city"].value_counts()
        if vc.iloc[0] >= 2 and vc.iloc[0] / len(g) >= 0.8:
            out[adj] = vc.index[0]
    out.update({"московский": "Москва", "санкт": "Санкт-Петербург", "ленинградский": "Санкт-Петербург",
                "петербургский": "Санкт-Петербург"})
    return out


def city_in_text(key, cities_re):
    key = re.split(r"\b(?:им|имени)\b", key)[0]
    for city, rx in cities_re:
        if rx.search(key):
            return city
    return None


def build_edu_map(cities):
    data = ROOT / "data"
    raw = pd.concat([pd.read_excel(data / f) for f in ("train.xlsx", "test.xlsx")])
    vals = pd.Series(raw[EDU_COLS].to_numpy().ravel()).dropna().astype(str)
    vals = vals[~vals.str.contains(NO_EDU, case=False, regex=False)]
    counts = vals.map(norm_edu).value_counts()
    counts = counts[counts.index != ""]
    keys = list(counts.index)

    wd, wd_names = wikidata_table(cities)
    hse, raex = hse_table(), raex_table()
    m_wd = match(keys, list(wd_names["key"]), FUZZ_MIN["wd"])
    m_hse = match(keys, list(hse["key"]), FUZZ_MIN["hse"])
    m_raex = match(keys, list(raex["key"]), FUZZ_MIN["raex"])
    wd_id = dict(zip(wd_names["key"], wd_names["id"]))
    id_names = wd_names.groupby("id")["key"].agg(list).to_dict()
    n_clash = 0
    for k, (wk, _) in list(m_wd.items()):
        if wk and eponym_clash(k, id_names[wd_id[wk]]):
            m_wd[k], n_clash = (None, 0.0), n_clash + 1
    for k, (hk, _) in list(m_hse.items()):
        if hk and eponym_clash(k, [hk]):
            m_hse[k], n_clash = (None, 0.0), n_clash + 1
    print(f"  dropped {n_clash} matches whose eponym (имени ...) disagrees")
    hse_i, raex_i = hse.set_index("key"), raex.set_index("key")
    hse_by_id, raex_by_id = {}, {}
    for src, table, by_id in [("hse", hse, hse_by_id), ("raex", raex, raex_by_id)]:
        links = match(list(table["key"]), list(wd_names["key"]), FUZZ_MIN[src])
        for k, (wk, _) in links.items():
            if wk and not (src == "hse" and hse_i.loc[k, "filial"]):
                by_id.setdefault(wd_id[wk], k)
        print(f"  {src}: {len(by_id)} rows linked to a Wikidata item")
    adj = adjective_city_map(wd, wd_names)
    big = cities[cities["population"] >= 50000]
    cities_re = city_regex(big["city"])

    rows = []
    for k in keys:
        r = {"key": k, "n_rows": int(counts[k])}
        wk, ws = m_wd[k]
        if wk:
            it = wd.loc[wd_id[wk]]
            r.update(wd_id=wd_id[wk], wd_score=ws, wd_label=it["label"], wd_country=it["country"],
                     wd_inception=it["inception"], wd_students=it["students"], wd_city=it["city"])
        hk, hs = m_hse[k]
        if not hk and r.get("wd_id") in hse_by_id:
            hk, hs = hse_by_id[r["wd_id"]], -1.0
        if hk:
            h = hse_i.loc[hk]
            r.update(hse_name=h["name"], hse_score=hs, ege_budget=h["ege_budget"], ege_paid=h["ege_paid"], hse_city=h["city"])
        rk, rs = m_raex[k]
        if not rk and r.get("wd_id") in raex_by_id:
            rk, rs = raex_by_id[r["wd_id"]], -1.0
        if rk:
            r.update(raex_score=rs, raex_rank=raex_i.loc[rk, "raex_rank"])
        first = k.split()[0] if k else ""
        explicit = city_in_text(k, cities_re)
        cands = [("filial_text", explicit if "филиал" in k else None),
                 ("text", explicit),
                 ("hse", r.get("hse_city")),
                 ("wikidata", r.get("wd_city")),
                 ("adjective", adj.get(first))]
        for src, c in cands:
            if isinstance(c, str) and c:
                r["edu_city"], r["city_source"] = c, src
                break
        rows.append(r)
    m = pd.DataFrame(rows)
    m.to_csv(OUT / "ext_edu_map.csv", index=False)
    w = m["n_rows"]
    for col in ["wd_id", "hse_name", "raex_rank", "edu_city"]:
        print(f"  {col:<10} covers {w[m[col].notna()].sum() / w.sum():.3f} of education mentions "
              f"({m[col].notna().sum()} of {len(m)} distinct strings)")
    print("  city source:", m.groupby("city_source")["n_rows"].sum().to_dict())
    return m


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    cities = build_cities()
    build_metro()
    build_edu_map(cities)
