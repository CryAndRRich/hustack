import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "mean_price"
ID = "index"
RAW_ID = "Unnamed: 0"
NAME = "ФИО"
LOW_PRICE = 1.0
SCALE_BUG = 100.0
NO_DESC = "не предоставил о себе"
NO_EDU = "не предоставил информацию об образовании"
STATUS = {
    "st_private": "Частный преподаватель",
    "st_student": "Студент",
    "st_school": "Школьный преподаватель",
    "st_univ": "Университетский преподаватель",
    "st_postgrad": "Аспирант или ординатор очной формы обучения",
}
LEVELS = {
    "cat_preschool": "Дошкольники",
    "cat_g1_4": "Школьники с 1 по 4 класс",
    "cat_g5_9": "Школьники с 5 по 9 класс",
    "cat_g10_11": "Школьники с 10 по 11 класс, студенты колледжей и техникумов, абитуриенты",
    "cat_univ": "Студенты университетов",
    "cat_adults": "Взрослые кроме абитуриентов и студентов",
}
TAGS = {
    "tag_school": "Школьная программа",
    "tag_ege": "Подготовка к ЕГЭ",
    "tag_oge": "Подготовка к ОГЭ",
    "tag_olymp": "Подготовка к олимпиадам",
}
EDU_COLS = [f"Education_{i}" for i in range(1, 7)]
EDU_DESC_COLS = [f"Desc_Education_{i}" for i in range(1, 7)]
DEGREE_COLS = ["Ученая степень 1", "Ученая степень 2"]
TITLE_COLS = ["Ученое звание 1", "Ученое звание 2"]
CAT_COLS = ["subject", "edu1"]


def get_data_dir():
    kaggle = Path("/kaggle/input")
    if kaggle.exists():
        hits = sorted(kaggle.rglob("train.xlsx")) or sorted(kaggle.rglob("train.csv"))
        if not hits:
            raise FileNotFoundError("no train.xlsx / train.csv under /kaggle/input")
        return hits[0].parent
    try:
        return Path(__file__).resolve().parent.parent / "data"
    except NameError:
        return Path.cwd() / "data"


def get_out_dir():
    return Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")


def xlsx_to_csv(src_dir, dst_dir):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for stem in ("train", "test"):
        csv = dst_dir / f"{stem}.csv"
        if not csv.exists():
            pd.read_excel(src_dir / f"{stem}.xlsx").to_csv(csv, index=False)
            print(f"converted {stem}.xlsx -> {csv}")
    return dst_dir


def load_data():
    d = get_data_dir()
    print(f"data dir: {d}")
    if (d / "train.xlsx").exists():
        d = xlsx_to_csv(d, get_out_dir() / "csv")
    train, test = pd.read_csv(d / "train.csv"), pd.read_csv(d / "test.csv")
    test = test.rename(columns={RAW_ID: ID})
    train = train.drop(columns=[RAW_ID])
    assert (test[ID].to_numpy() == np.arange(len(test))).all(), "test index is not 0..n-1"
    return train, test


def restore_price(y):
    y = np.asarray(y, dtype=float)
    low = y < LOW_PRICE
    return np.where(low, y * SCALE_BUG, y), low


def parse_list(s):
    if not isinstance(s, str) or not s.strip():
        return []
    try:
        v = ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return []
    return [str(t).strip() for t in v] if isinstance(v, list) else []


def norm_text(s):
    return re.sub(r"\s+", " ", str(s)).strip().lower() if isinstance(s, str) else ""


EDU_ABBR = [
    (r"\bун-т\w*", "университет"), (r"\bин-т\w*", "институт"), (r"\bгос\b", "государственный"),
    (r"\bтехн\b", "технический"), (r"\bтехнол\b", "технологический"), (r"\bполитехн\b", "политехнический"),
    (r"\bпед\b", "педагогический"), (r"\bмоск\b", "московский"), (r"\bакад\b", "академия"),
    (r"\bнац\b", "национальный"), (r"\bмед\b", "медицинский"), (r"\bгу\b", "государственный университет"),
]
EDU_NOISE = r"\b(с отличием|красный диплом|фгбоу|фгаоу|фгоу|гоувпо|гоу|гбоу|впо|во|ноу|чоу|ано|оу)\b"


def norm_edu(s):
    if not isinstance(s, str):
        return ""
    s = s.lower().replace("ё", "е").replace("\xa0", " ")
    s = re.sub(r"[«»\"'“”„()\[\]]", " ", s)
    s = s.replace(".", " ")
    for pat, rep in EDU_ABBR:
        s = re.sub(pat, rep, s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(EDU_NOISE, " ", s)
    return re.sub(r"\s+", " ", s).strip()


CITY_FORM_FIX = {"Крымск": "Симферополь", "Донской": "Ростов-на-Дону"}


def city_forms(city):
    n = norm_edu(city)
    forms = {n}
    if " " in n:
        return forms
    if n[-1] in "ая":
        forms |= {n[:-1] + e for e in ("ы", "и", "е", "у", "ой", "ей")}
    elif n[-1] == "ь":
        forms |= {n[:-1] + e for e in ("и", "ю", "ем", "я", "е", "ский", "ская", "ского", "ском")}
    elif n.endswith("ск"):
        forms |= {n + e for e in ("а", "е", "у", "ом", "ий", "ая", "ое", "ого", "ом", "ой", "их")}
    elif n[-1] not in "аеиоуыэюяй":
        forms |= {n + e for e in ("а", "е", "у", "ом", "ский", "ская", "ское", "ского", "ском")}
    return forms


def city_regex(cities):
    pats = [(CITY_FORM_FIX.get(c, c), re.compile(r"\b(?:" + "|".join(sorted(map(re.escape, city_forms(c)))) + r")\b"))
            for c in cities]
    return pats


def fill_rating_by_name(train, test):
    both = pd.concat([train[[NAME, "tutor_rating"]], test[[NAME, "tutor_rating"]]])
    by_name = both.dropna().groupby(NAME)["tutor_rating"].median()
    for df in (train, test):
        df["tutor_rating"] = df["tutor_rating"].fillna(df[NAME].map(by_name))


def build_features(df):
    out = pd.DataFrame(index=df.index)
    out["subject"] = df["предмет"].fillna("missing").astype(str)

    status = df["status"].fillna("")
    for k, v in STATUS.items():
        out[k] = status.str.contains(v, regex=False).astype(int)
    out["n_status"] = out[list(STATUS)].sum(axis=1)

    out["experience"] = pd.to_numeric(df["experience"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    out["rating"] = df["tutor_rating"].astype(float)
    out["reviews"] = df["tutor_reviews"].astype(float)
    out["log_reviews"] = np.log1p(out["reviews"])

    tags = df["tutor_head_tags"].map(parse_list)
    for k, v in TAGS.items():
        out[k] = tags.map(lambda t, v=v: int(v in t))
    out["n_tags"] = tags.map(len)
    out["n_tags_higher"] = out["n_tags"] - out[list(TAGS)].sum(axis=1)

    cats = df["categories"].map(parse_list)
    for k, v in LEVELS.items():
        out[k] = cats.map(lambda c, v=v: int(any(x == v or x.startswith(v + " ") for x in c)))
    out["n_levels"] = out[list(LEVELS)].sum(axis=1)
    out["n_pair"] = cats.map(lambda c: sum(x.endswith(" в паре") for x in c))
    out["n_group"] = cats.map(lambda c: sum(x.endswith(" группа") for x in c))

    desc = df["description"].map(norm_text)
    out["desc_len"] = desc.str.len()
    out["desc_words"] = desc.str.split().map(len)
    out["desc_empty"] = desc.str.contains(NO_DESC, regex=False).astype(int)
    exp = df["experience_desc"].map(norm_text)
    out["expdesc_len"] = exp.str.len()

    edu = df[EDU_COLS].apply(lambda c: c.map(norm_text))
    out["edu_missing"] = edu["Education_1"].str.contains(NO_EDU, regex=False).astype(int)
    out["n_edu"] = (edu != "").sum(axis=1) - out["edu_missing"]
    out["edu1"] = edu["Education_1"].where(edu["Education_1"] != "", "missing")
    years = df[EDU_DESC_COLS].apply(
        lambda c: pd.to_numeric(c.astype(str).str.extract(r"Год окончания:\s*(\d{4})")[0], errors="coerce"))
    out["grad_year_first"] = years.min(axis=1)
    out["grad_year_last"] = years.max(axis=1)

    degree = df[DEGREE_COLS].apply(lambda c: c.map(norm_text)).agg(" ".join, axis=1)
    title = df[TITLE_COLS].apply(lambda c: c.map(norm_text)).agg(" ".join, axis=1)
    out["deg_candidate"] = degree.str.contains("кандидат", regex=False).astype(int)
    out["deg_doctor"] = degree.str.contains("доктор", regex=False).astype(int)
    out["title_docent"] = title.str.contains("доцент", regex=False).astype(int)
    out["title_prof"] = title.str.contains("профессор", regex=False).astype(int)
    out["title_researcher"] = title.str.contains("научный сотрудник", regex=False).astype(int)
    return out


EXT_EDU, EXT_CITIES, EXT_METRO = "ext_edu_map.csv", "ext_cities.csv", "ext_metro.csv"
TEXT_CITY_MIN_POP = 100_000
METRO_WEIGHT = 2
CITY_EXTRA = {
    "Москва": r"\b(?:москв\w*|московск\w*|мск|подмосковь\w*|мкад)\b",
    "Санкт-Петербург": r"\b(?:санкт петербург\w*|петербург\w*|спб|питер\w*|ленинград\w*)\b",
    "Нижний Новгород": r"\b(?:нижн\w* новгород\w*|нижегородск\w*)\b",
    "Ростов-на-Дону": r"\b(?:ростов\w* на дону|ростовск\w*)\b",
}
METRO_RE = re.compile(r"(?:\bметро|\bст\.?\s*м\.?|\bм\.)\s*[«\"]?([А-ЯЁ][а-яё\-]+(?:\s[А-ЯЁ]?[а-яё\-]+)?)")
EXT_CAT_COLS = ["e1_city", "loc_city", "loc_fd"]
MSK_SCHOOL_RE = re.compile(r"(?:школ\w*|гимнази\w*|лице\w*|сош|гбоу|гбу|цо)[^.\n№]{0,30}№\s*(\d{4})\b", re.I)
MSK_SCHOOL_MIN, MSK_SCHOOL_MAX = 1000, 2200
FOREIGN_RE = (r"\b(?:харьков|киев|одесс|днепр|львов|запорож|минск|белорус|гомел|витебск|гродн|могилев|брест|"
              r"казах|алма|акмол|караганд|семипалат|астан|павлодар|ташкент|узбек|самарканд|тбилис|грузин|"
              r"ереван|армян|бакин|азербайдж|кишин|молдав|рижск|латвий|вильн|литов|таллин|эстон|бишкек|"
              r"киргиз|кыргыз|душанб|таджик|ашхабад|туркмен)\w*")


def find_external():
    roots = [Path("/kaggle/input")] if Path("/kaggle/input").exists() else [get_data_dir() / "external"]
    for root in roots:
        hits = sorted(root.rglob(EXT_EDU))
        if hits:
            return hits[0].parent
    raise FileNotFoundError(f"{EXT_EDU} not found: attach the external dataset (data/data.zip)")


def load_external():
    d = find_external()
    print(f"external dir: {d}")
    edu = pd.read_csv(d / EXT_EDU, dtype={"key": str}).dropna(subset=["key"]).set_index("key")
    cities = pd.read_csv(d / EXT_CITIES).set_index("city")
    missing = [c for c in WAGE_V3_COLS if c not in cities.columns]
    if missing:
        raise RuntimeError(f"{EXT_CITIES} lacks {missing}: re-upload the V3 data.zip to the external dataset")
    metro = pd.read_csv(d / EXT_METRO)
    metro["station"] = metro["station"].str.lower().str.replace("ё", "е")
    amb = metro.groupby("station")["city"].nunique()
    metro = metro[metro["station"].isin(amb[amb == 1].index)].drop_duplicates("station")
    print(f"external: {len(edu)} education keys, {len(cities)} cities, {len(metro)} unambiguous metro stations")
    return {"edu": edu, "cities": cities, "metro": metro.set_index("station")["city"]}


def city_attrs(city, cities, prefix):
    out = pd.DataFrame(index=city.index)
    out[f"{prefix}_msk"] = (city == "Москва").astype(int)
    out[f"{prefix}_spb"] = (city == "Санкт-Петербург").astype(int)
    out[f"{prefix}_log_pop"] = np.log(city.map(cities["population"]).astype(float))
    out[f"{prefix}_wage"] = city.map(cities["region_wage"]).astype(float)
    out[f"{prefix}_grp"] = city.map(cities["region_grp_pc"]).astype(float)
    return out


def edu_features(df, ext):
    edu, cities = ext["edu"], ext["cities"]
    keys = df[EDU_COLS].apply(lambda c: c.map(norm_edu))
    look = {f: keys.apply(lambda c, f=f: c.map(edu[f])) for f in ("edu_city", "ege_budget", "ege_paid", "raex_rank")}
    first = EDU_COLS[0]
    e1_city = look["edu_city"][first]
    out = city_attrs(e1_city, cities, "e1")
    out["e1_city"] = e1_city.fillna("missing")
    out["e1_matched"] = keys[first].map(edu["wd_id"]).notna().astype(int)
    out["e1_inception"] = keys[first].map(edu["wd_inception"]).astype(float)
    out["e1_log_students"] = np.log1p(keys[first].map(edu["wd_students"]).astype(float))
    out["e1_ege_budget"] = look["ege_budget"][first].astype(float)
    out["e1_ege_paid"] = look["ege_paid"][first].astype(float)
    out["e1_raex_rank"] = look["raex_rank"][first].astype(float)
    out["edu_any_msk"] = (look["edu_city"] == "Москва").any(axis=1).astype(int)
    out["edu_any_spb"] = (look["edu_city"] == "Санкт-Петербург").any(axis=1).astype(int)
    out["edu_n_cities"] = look["edu_city"].nunique(axis=1)
    out["edu_max_ege"] = look["ege_budget"].astype(float).max(axis=1)
    out["edu_best_raex"] = look["raex_rank"].astype(float).min(axis=1)
    return out


def detect_city(raw, ext, use_metro, schools):
    cities, metro = ext["cities"], ext["metro"]
    low = raw.map(norm_edu)
    big = cities.index[cities["population"] >= TEXT_CITY_MIN_POP]
    pats = [(c, re.compile(CITY_EXTRA[c])) for c in CITY_EXTRA]
    pats += [(c, rx) for c, rx in city_regex([c for c in big if c not in CITY_EXTRA])]
    counts = pd.DataFrame(0, index=raw.index, columns=sorted({c for c, _ in pats}), dtype=float)
    for c, rx in pats:
        counts[c] += low.map(lambda s, rx=rx: len(rx.findall(s)))
    metro_city = pd.Series(None, index=raw.index, dtype=object)
    if use_metro:
        stations = raw.map(lambda s: [m.lower().replace("ё", "е").strip() for m in METRO_RE.findall(s)])
        metro_city = stations.map(lambda st: next((metro[s] for s in st if s in metro.index), None))
        for i, c in metro_city.dropna().items():
            if c in counts.columns:
                counts.loc[i, c] += METRO_WEIGHT
    n_school = pd.Series(0, index=raw.index)
    if schools:
        n_school = raw.map(lambda s: sum(MSK_SCHOOL_MIN <= int(n) <= MSK_SCHOOL_MAX for n in MSK_SCHOOL_RE.findall(s)))
        counts["Москва"] += METRO_WEIGHT * n_school
    pop = cities["population"].reindex(counts.columns).fillna(0).to_numpy()
    score = counts.to_numpy() + pop / pop.max() * 1e-3
    city = pd.Series(np.where(counts.to_numpy().sum(axis=1) > 0, counts.columns[score.argmax(axis=1)], None),
                     index=raw.index)
    return city, counts, metro_city, n_school


def text_city(df, ext, schools=False):
    raw = (df["description"].fillna("").astype(str) + " " + df["experience_desc"].fillna("").astype(str))
    city, counts, metro_city, n_school = detect_city(raw, ext, use_metro=True, schools=schools)
    out = city_attrs(city, ext["cities"], "txt")
    out["txt_n_cities"] = (counts > 0).sum(axis=1)
    out["txt_metro"] = metro_city.notna().astype(int)
    if schools:
        out["txt_msk_school"] = (n_school > 0).astype(int)
    return out, city


def edu_raw(df):
    return df[EDU_COLS + EDU_DESC_COLS].fillna("").astype(str).agg(" ".join, axis=1)


def ext_features(df, ext, block_a=False):
    e = edu_features(df, ext)
    t, txt_city = text_city(df, ext, schools=block_a)
    e1 = e["e1_city"].replace("missing", np.nan)
    if block_a:
        edu_city = detect_city(edu_raw(df), ext, use_metro=False, schools=True)[0]
        fill = e1.isna() & edu_city.notna()
        e1 = e1.where(~fill, edu_city)
        for col, v in city_attrs(e1, ext["cities"], "e1").items():
            e[col] = v
        e["e1_city"] = e1.fillna("missing")
        e["e1_city_from_text"] = fill.astype(int)
        edu_low = df[EDU_COLS].fillna("").astype(str).apply(lambda c: c.map(norm_edu))
        foreign = edu_low.apply(lambda c: c.str.contains(FOREIGN_RE) | (c.str.contains(r"[a-z]{3}") & ~c.str.contains(r"[а-я]")))
        e["e1_foreign"] = foreign[EDU_COLS[0]].astype(int)
        e["edu_any_foreign"] = foreign.any(axis=1).astype(int)
    loc = txt_city.where(txt_city.notna(), e1)
    out = pd.concat([e, t, city_attrs(loc, ext["cities"], "loc")], axis=1)
    out["loc_city"] = loc.fillna("missing")
    out["loc_fd"] = loc.map(ext["cities"]["federal_district"]).fillna("missing")
    out["loc_source"] = np.select([txt_city.notna(), loc.notna()], [2, 1], 0)
    if block_a:
        out.loc[(out["loc_source"] == 0) & (out["edu_any_foreign"] == 1), "loc_fd"] = "foreign"
    return out


WAGE_V3_COLS = ["region_teacher_wage", "region_addl_teacher_wage", "region_univ_wage", "region_wage_2023"]


def teacher_wage_features(E, ext):
    c = ext["cities"]
    out = pd.DataFrame(index=E.index)
    for pre in ("loc", "e1"):
        city = E[f"{pre}_city"].replace("missing", np.nan)
        for col, short in [("region_teacher_wage", "teacher_wage"), ("region_addl_teacher_wage", "addl_teacher_wage"),
                           ("region_univ_wage", "univ_wage")]:
            out[f"{pre}_{short}"] = city.map(c[col]).astype(float)
        out[f"{pre}_teacher_ratio"] = out[f"{pre}_teacher_wage"] / city.map(c["region_wage_2023"]).astype(float)
    return out


CITY_NUM_COLS = [f"{p}_{s}" for p in ("e1", "txt", "loc") for s in ("log_pop", "wage", "grp")]


def fill_city_numeric(Xtr, Xte):
    Xtr, Xte = Xtr.copy(), Xte.copy()
    for prefix in ("e1", "txt", "loc"):
        known = Xtr[f"{prefix}_wage"].notna()
        Xtr[f"{prefix}_known"], Xte[f"{prefix}_known"] = known.astype(int), Xte[f"{prefix}_wage"].notna().astype(int)
    for c in CITY_NUM_COLS:
        med = Xtr[c].median()
        Xtr[c], Xte[c] = Xtr[c].fillna(med), Xte[c].fillna(med)
    return Xtr, Xte


PRICE_NUM = r"(\d{1,2}\s?\d{3}|\d{3,5})"
PRICE_RUB = r"\s*(?:руб\w*\.?|р\.|р\b|₽)"
PRICE_DUR = r"(\d{2,3})\s*(?:мин\w*\.?)|(\d(?:[.,]\d)?)?\s*(?:ак\w*\.?\s*)?час\w*"
PRICE_AFTER = re.compile(PRICE_NUM + PRICE_RUB + r"\s*(?:\.|,)?\s*(?:/|за|в|-|—|–)?\s*(?:" + PRICE_DUR + r")", re.I)
PRICE_BEFORE = re.compile(r"(?:(\d{2,3})\s*мин\w*\.?|(\d(?:[.,]\d)?)\s*(?:ак\w*\.?\s*)?час\w*)\s*[-—–:]?\s*(?:-|—|–)?\s*"
                          + PRICE_NUM + PRICE_RUB, re.I)
PRICE_ANY = re.compile(PRICE_NUM + PRICE_RUB, re.I)
PRICE_NOT_LESSON = re.compile(r"мес|курс|абонемент|пакет|вебинар|с человека|скидк|бесплатн", re.I)


def _minutes(m_min, m_hours, text):
    if m_min:
        return float(m_min)
    h = float(m_hours.replace(",", ".")) if m_hours else 1.0
    return (45.0 if re.search(r"ак\w*\.?\s*час", text, re.I) else 60.0) * h


def parse_prices(text):
    t = str(text).replace("\xa0", " ")
    ppm, spans = [], []
    for m in PRICE_AFTER.finditer(t):
        n, d = float(m.group(1).replace(" ", "")), _minutes(m.group(2), m.group(3), m.group(0))
        if 30 <= d <= 180 and 100 <= n <= 20000:
            ppm.append(n / d)
            spans.append(m.span())
    for m in PRICE_BEFORE.finditer(t):
        n, d = float(m.group(3).replace(" ", "")), _minutes(m.group(1), m.group(2), m.group(0))
        if 30 <= d <= 180 and 100 <= n <= 20000 and not any(a <= m.start(3) < b for a, b in spans):
            ppm.append(n / d)
            spans.append(m.span())
    bare = []
    for m in PRICE_ANY.finditer(t):
        if any(a <= m.start() < b for a, b in spans):
            continue
        n = float(m.group(1).replace(" ", ""))
        if 300 <= n <= 5000 and not PRICE_NOT_LESSON.search(t[max(0, m.start() - 40): m.end() + 25]):
            bare.append(n)
    return ppm, bare


def price_features(df):
    raw = (df["description"].fillna("").astype(str) + " || "
           + df["experience_desc"].fillna("").astype(str).str.replace("_x000D_", " ", regex=False))
    res = raw.map(parse_prices)
    out = pd.DataFrame(index=df.index)
    out["txtp_ppm_med"] = res.map(lambda r: float(np.median(r[0])) if r[0] else np.nan)
    out["txtp_ppm_min"] = res.map(lambda r: min(r[0]) if r[0] else np.nan)
    out["txtp_ppm_max"] = res.map(lambda r: max(r[0]) if r[0] else np.nan)
    out["txtp_n"] = res.map(lambda r: len(r[0]))
    out["txtp_bare_med"] = res.map(lambda r: float(np.median(r[1])) / 60 if r[1] else np.nan)
    return out


REF_YEAR = 2025
KEYWORDS = {
    "kw_online": r"онлайн|skype|скайп|zoom|зум|discord|дистанц|удаленн",
    "kw_at_home": r"у себя|у меня дома|на дому|ко мне",
    "kw_travel": r"выезд|выезжаю|к ученику|у ученика",
    "kw_expert": r"эксперт",
    "kw_high_cat": r"высш\w* (?:квалификационн\w* )?категори",
    "kw_olymp_win": r"победител|призер",
    "kw_hundred": r"100 балл|стобалльн|сто балл",
    "kw_author": r"автор|методист",
    "kw_degree": r"кандидат\w* (?:физ|тех|пед|наук)|к\.\s?ф\.?-?м\.?\s?н|к\.\s?т\.\s?н",
    "kw_univ_teach": r"кафедр|доцент|преподава\w* (?:в|на) (?:вуз|университет|институт)",
    "kw_higher_math": r"высш\w* математ|мат\w* анализ|линейн\w* алгебр",
    "kw_python": r"python|питон",
    "kw_cpp": r"c\+\+|с\+\+",
    "kw_java": r"\bjava\b",
    "kw_pascal": r"pascal|паскаль",
    "kw_profil": r"профил",
    "kw_school": r"учител|школ",
    "kw_guarantee": r"гарант",
    "kw_trial": r"пробн\w* (?:урок|заняти)",
}


def extra_features(df, name_counts):
    out = pd.DataFrame(index=df.index)
    desc = df["description"].fillna("").astype(str)
    exp = df["experience_desc"].fillna("").astype(str).str.replace("_x000D_", " ", regex=False)
    low = (desc + " " + exp).str.lower().str.replace("ё", "е")
    for k, pat in KEYWORDS.items():
        out[k] = low.str.contains(pat).astype(int)
    out["kw_n"] = out[list(KEYWORDS)].sum(axis=1)

    last = df[NAME].str.extract(r"(\S+)\s*-\s*фотография\s*$")[0].fillna("").str.lower()
    out["male"] = np.select([last.str.endswith("ич"), last.str.endswith(("вна", "чна", "кызы"))], [1, 0], -1)
    out["n_copies"] = df[NAME].map(name_counts).fillna(1).astype(int)

    years = df[EDU_DESC_COLS].apply(
        lambda c: pd.to_numeric(c.astype(str).str.extract(r"Год окончания:\s*(\d{4})")[0], errors="coerce"))
    out["age_proxy"] = REF_YEAR - years.min(axis=1) + 22
    out["years_since_grad"] = REF_YEAR - years.max(axis=1)
    exp_years = pd.to_numeric(df["experience"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    out["reviews_per_year"] = df["tutor_reviews"].astype(float) / exp_years.clip(lower=1)
    out["rating_x_reviews"] = df["tutor_rating"].astype(float) * np.log1p(df["tutor_reviews"].astype(float))
    cats = df["categories"].map(parse_list)
    n_pg = cats.map(lambda c: sum(x.endswith((" в паре", " группа")) for x in c))
    out["pair_group_share"] = n_pg / cats.map(len).clip(lower=1)

    yrs = exp.str.findall(r"\b(19[6-9]\d|20[0-2]\d)\b")
    out["exp_n_years"] = yrs.map(len)
    out["exp_first_year"] = yrs.map(lambda v: min(map(int, v)) if v else np.nan)
    out["exp_to_present"] = exp.str.lower().str.contains(r"по настоящее|по н\.\s?в|по сей день|настоящее время").astype(int)

    out["desc_sentences"] = desc.str.count(r"[.!?]+")
    out["desc_digit_share"] = desc.str.count(r"\d") / desc.str.len().clip(lower=1)
    out["desc_latin_share"] = desc.str.count(r"[A-Za-z]") / desc.str.len().clip(lower=1)
    out["desc_exclaim"] = desc.str.count("!")
    return out
