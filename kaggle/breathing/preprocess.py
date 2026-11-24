import numpy as np
import pandas as pd
from pathlib import Path

CONSUMPTION = ["coal_consumption", "oil_consumption", "gas_consumption",
               "nuclear_consumption", "renewables_consumption"]
PRODUCTION = ["oil_production", "gas_production", "coal_production"]
ELEC_SOURCES = ["coal_electricity", "gas_electricity", "oil_electricity",
                "nuclear_electricity", "hydro_electricity", "wind_electricity",
                "solar_electricity"]
FOSSIL3 = ["coal_consumption", "oil_consumption", "gas_consumption"]

K0 = 0.22
KWH_PER_KGOE = 11.63

PHYS_TERMS = ["coal", "oil", "gas", "bunker", "flared", "otherref"]
PHYS_SIGN = np.array([1.0, 1.0, 1.0, -1.0, 1.0, -1.0])
PHYS_INIT = np.array([0.349, 0.263, 0.205, 0.263, 0.205, 0.1315])

UN_TERMS = ["un_solid_twh", "un_liquid_twh", "un_gas_twh", "un_nonenergy_twh", "un_flared_twh"]
UN_SIGN = np.array([1.0, 1.0, 1.0, -1.0, 1.0])
UN_INIT = np.array([0.349, 0.263, 0.205, 0.263, 0.205])


def find_data_dir():
    for p in Path("/kaggle/input").glob("*"):
        if (p / "train.csv").exists():
            return p
    try:
        local = Path(__file__).resolve().parent.parent / "data"
    except NameError:
        local = Path.cwd() / "data"
    return local if (local / "train.csv").exists() else Path.cwd() / "data"


def find_ext():
    for p in [find_data_dir() / "ext_features.csv", *Path("/kaggle/input").glob("*/ext_features.csv")]:
        if p.exists():
            return p
    raise FileNotFoundError("ext_features.csv (built by src/build_ext.py) not found")


def load():
    d = find_data_dir()
    ext = pd.read_csv(find_ext())
    tr = pd.read_csv(d / "train.csv").merge(ext, on=["country", "year"], how="left")
    te = pd.read_csv(d / "test.csv").merge(ext, on=["country", "year"], how="left")
    return tr, te


def fossil_energy(df):
    have3 = df[FOSSIL3].notna().all(axis=1)
    pe = df["primary_energy_consumption"]
    has_un = df["un_liquid_twh"].notna()
    route = np.where(have3, 0, np.where(pe.notna(), 1, np.where(has_un, 3, 2)))
    e = np.where(have3, df[FOSSIL3].sum(axis=1), pe)
    e = np.where(route >= 2, np.nan, e)
    return pd.Series(e, index=df.index), pd.Series(route, index=df.index)


def un_log(df, k):
    lin = df[UN_TERMS].fillna(0.0).to_numpy() @ (UN_SIGN * k)
    return np.where(df["un_liquid_twh"].notna(), np.log1p(np.maximum(lin, 1e-3)), np.nan)


def phys_terms(df, route):
    host = df[FOSSIL3].to_numpy()
    eia = df[["eia_coal_twh", "eia_oil_twh", "eia_gas_twh"]].to_numpy()
    has_eia = ~np.isnan(eia).any(axis=1)
    r = route.to_numpy()
    fuels = np.where((r == 0)[:, None], host, np.where(has_eia[:, None], eia, 0.0))
    extra = df[["eia_bunker_twh", "eia_flared_twh", "eia_otherref_twh"]].fillna(0.0).to_numpy()
    T = np.hstack([np.nan_to_num(fuels), extra])
    T[r >= 2] = 0.0
    fallback = np.where((r == 1) & ~has_eia, df["primary_energy_consumption"].to_numpy(), 0.0)
    out = pd.DataFrame(T, columns=PHYS_TERMS, index=df.index)
    out["pe_fallback"] = np.nan_to_num(fallback)
    return out


def phys_log(T, k):
    lin = T[PHYS_TERMS].to_numpy() @ (PHYS_SIGN * k) + K0 * T["pe_fallback"].to_numpy()
    return np.log1p(np.maximum(lin, 1e-3))


def _log_ratio(num, den, lo=-12.0, hi=6.0):
    out = np.log((num + 1.0) / den.where(den > 0))
    return out.clip(lo, hi)


def _wdi_features(df, e=None):
    f = pd.DataFrame(index=df.index)
    pop = df["population"]
    gdp_pc = df["wb_gdp_pc_ppp"].fillna(df["wb_gdp_pc_usd"])
    f["wb_log_gdp_pc"] = np.log(gdp_pc)
    for c in ["wb_urban_pct", "wb_renew_final_pct", "wb_elec_access_pct", "wb_fossil_pct",
              "wb_elec_fossil_pct", "wb_industry_pct_gdp", "wb_oil_rents_pct", "wb_gas_rents_pct",
              "wb_coal_rents_pct", "wb_fuel_imports_pct", "wb_fuel_exports_pct"]:
        f[c] = df[c] / 100.0
    f["wb_log_density"] = np.log(df["wb_pop_density"])
    f["wb_log_energy_pc"] = np.log(df["wb_energy_pc_kgoe"])
    for c in ["wb_container_teu", "wb_air_pax", "wb_tourist_arrivals"]:
        f[c.replace("wb_", "wb_log_") + "_pc"] = np.log((df[c] + 1.0) / pop)
    if e is not None:
        f["wb_energy_over_e"] = _log_ratio(df["wb_energy_pc_kgoe"] * pop * KWH_PER_KGOE / 1e9, e)
        f["wb_log_teu_per_e"] = _log_ratio(df["wb_container_teu"], e * 1e6)
        f["biofuel_share"] = (df["biofuel_consumption"] / e.where(e > 0)).clip(0.0, 1.0)
    return f


def _eia_features(df, e):
    f = pd.DataFrame(index=df.index)
    parts = ["eia_coal_twh", "eia_oil_twh", "eia_gas_twh", "eia_nuclear_twh", "eia_renew_twh"]
    tot = df[parts].sum(axis=1, min_count=5)
    for c in parts:
        f["eia_sh_" + c[4:-4]] = df[c] / tot.where(tot > 0)
    oil = df["eia_oil_twh"]
    fossil = df[["eia_coal_twh", "eia_oil_twh", "eia_gas_twh"]].sum(axis=1, min_count=3)
    f["eia_bunker_per_oil"] = (df["eia_bunker_twh"] / oil.where(oil > 0)).clip(0.0, 2.0)
    f["eia_otherref_per_oil"] = (df["eia_otherref_twh"] / oil.where(oil > 0)).clip(0.0, 2.0)
    f["eia_flared_per_fossil"] = (df["eia_flared_twh"] / fossil.where(fossil > 0)).clip(0.0, 5.0)
    f["eia_fossil_over_e"] = _log_ratio(fossil, e)
    f["eia_bunker_imputed"] = df["eia_bunker_twh_imputed"]
    f["eia_flared_imputed"] = df["eia_flared_twh_imputed"]
    return f


def _un_features(df, e, T):
    f = pd.DataFrame(index=df.index)
    tot = df[["un_solid_twh", "un_liquid_twh", "un_gas_twh"]].sum(axis=1, min_count=3)
    for c in ["un_solid_twh", "un_liquid_twh", "un_gas_twh"]:
        f["un_sh_" + c[3:-4]] = df[c] / tot.where(tot > 0)
    f["un_nonenergy_per_fossil"] = (df["un_nonenergy_twh"] / tot.where(tot > 0)).clip(0.0, 2.0)
    f["un_flared_per_fossil"] = (df["un_flared_twh"] / tot.where(tot > 0)).clip(0.0, 5.0)
    f["un_fossil_over_e"] = _log_ratio(tot, e)
    v3 = pd.Series(phys_log(T, PHYS_INIT), index=df.index).where(T[PHYS_TERMS[:3]].sum(axis=1) > 0)
    f["un_minus_v3"] = (pd.Series(un_log(df, UN_INIT), index=df.index) - v3).clip(-3.0, 3.0)
    return f


def build_features(df, e, route, T):
    f = pd.DataFrame(index=df.index)
    pop = df["population"]
    eg = df["electricity_generation"]
    for c in PRODUCTION:
        f["lr_" + c] = _log_ratio(df[c], e)
    f["elec_per_energy"] = (eg / e.where(e > 0)).clip(upper=5.0)
    for c in ELEC_SOURCES:
        f["sh_" + c] = (df[c] / eg.where(eg > 0)).clip(0.0, 1.5)
    f["netimp_share"] = (df["net_elec_imports"] / eg.where(eg > 0)).clip(-3.0, 3.0)
    f["log_energy_per_cap"] = np.log(e.where(e > 0) / pop * 1e6 + 1.0)
    f["log_gdp_per_cap"] = np.log(df["gdp"] / pop)
    f["log_pop"] = np.log(pop)
    f["year_c"] = (df["year"] - 2011.0) / 11.0
    for c in CONSUMPTION:
        f["mix_" + c] = (df[c] / e.where(e > 0)).clip(0.0, 1.5)
    f["route"] = route.astype(float)
    for c in CONSUMPTION + PRODUCTION + ELEC_SOURCES + ["gdp", "primary_energy_consumption",
                                                         "electricity_generation", "net_elec_imports"]:
        f["na_" + c] = df[c].isna().astype(float)
    cont = pd.get_dummies(df["continent"], prefix="cont", dtype=float)
    return pd.concat([f, cont, _wdi_features(df, e), _eia_features(df, e), _un_features(df, e, T)], axis=1)


def percap_features(df):
    f = _wdi_features(df)
    f["log_pop"] = np.log(df["population"])
    f["year_c"] = (df["year"] - 2011.0) / 11.0
    return pd.concat([f, pd.get_dummies(df["continent"], prefix="cont", dtype=float)], axis=1)


def prepare(tr, te):
    e_tr, r_tr = fossil_energy(tr)
    e_te, r_te = fossil_energy(te)
    T_tr, T_te = phys_terms(tr, r_tr), phys_terms(te, r_te)
    X_tr = build_features(tr, e_tr, r_tr, T_tr)
    X_te = build_features(te, e_te, r_te, T_te)
    X_te = X_te.reindex(columns=X_tr.columns, fill_value=0.0)
    P_tr = percap_features(tr)
    P_te = percap_features(te).reindex(columns=P_tr.columns, fill_value=0.0)
    return X_tr, X_te, P_tr, P_te, e_tr, e_te, r_tr, r_te, T_tr, T_te
