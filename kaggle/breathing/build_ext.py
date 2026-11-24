import json
import re
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
WDI = {
    "NY.GDP.PCAP.PP.KD": "wb_gdp_pc_ppp",
    "NY.GDP.PCAP.KD": "wb_gdp_pc_usd",
    "SP.URB.TOTL.IN.ZS": "wb_urban_pct",
    "AG.LND.TOTL.K2": "wb_land_km2",
    "EG.FEC.RNEW.ZS": "wb_renew_final_pct",
    "EG.ELC.ACCS.ZS": "wb_elec_access_pct",
    "EG.USE.PCAP.KG.OE": "wb_energy_pc_kgoe",
    "EG.USE.COMM.FO.ZS": "wb_fossil_pct",
    "EG.ELC.FOSL.ZS": "wb_elec_fossil_pct",
    "IS.SHP.GOOD.TU": "wb_container_teu",
    "IS.AIR.PSGR": "wb_air_pax",
    "ST.INT.ARVL": "wb_tourist_arrivals",
    "NV.IND.TOTL.ZS": "wb_industry_pct_gdp",
    "NY.GDP.PETR.RT.ZS": "wb_oil_rents_pct",
    "NY.GDP.NGAS.RT.ZS": "wb_gas_rents_pct",
    "NY.GDP.COAL.RT.ZS": "wb_coal_rents_pct",
    "EN.POP.DNST": "wb_pop_density",
    "TM.VAL.FUEL.ZS.UN": "wb_fuel_imports_pct",
    "TX.VAL.FUEL.ZS.UN": "wb_fuel_exports_pct",
}
ISO_EXTRA = {"Andorra": "AND", "Anguilla": "AIA", "Bonaire Sint Eustatius and Saba": "BES",
             "Liechtenstein": "LIE", "Marshall Islands": "MHL", "Palau": "PLW",
             "Sint Maarten (Dutch part)": "SXM", "Wallis and Futuna": "WLF"}
TWH_PER_QUAD = 293.071
TWH_PER_TBPD = 0.62
TWH_PER_BCM = 10.4
EIA = {r"INTL\.4411-2-([A-Z]{3})-QBTU\.A": ("eia_coal_twh", TWH_PER_QUAD),
       r"INTL\.4413-2-([A-Z]{3})-QBTU\.A": ("eia_gas_twh", TWH_PER_QUAD),
       r"INTL\.4415-2-([A-Z]{3})-QBTU\.A": ("eia_oil_twh", TWH_PER_QUAD),
       r"INTL\.4417-2-([A-Z]{3})-QBTU\.A": ("eia_nuclear_twh", TWH_PER_QUAD),
       r"INTL\.4418-2-([A-Z]{3})-QBTU\.A": ("eia_renew_twh", TWH_PER_QUAD),
       r"INTL\.54-13-([A-Z]{3})-TBPD\.A": ("eia_bunker_twh", TWH_PER_TBPD),
       r"INTL\.68-2-([A-Z]{3})-QBTU\.A": ("eia_otherref_twh", TWH_PER_QUAD),
       r"INTL\.43-1-([A-Z]{3})-BCM\.A": ("eia_flared_twh", TWH_PER_BCM)}


def fill_tail(df, col, base=None):
    df = df.sort_values(["iso3", "year"]).copy()
    v = df[col] if base is None else df[col] / df[base].where(df[base] > 0)
    last3 = v.dropna().groupby(df["iso3"]).apply(lambda s: s.tail(3).mean())
    fill = df["iso3"].map(last3) * (1.0 if base is None else df[base])
    df[col + "_imputed"] = (df[col].isna() & fill.notna()).astype(float)
    df[col] = df[col].fillna(fill)
    return df


def read_eia():
    pats = [(re.compile(p), name, k) for p, (name, k) in EIA.items()]
    rows = []
    with zipfile.ZipFile(DATA / "external/eia/INTL.zip") as z, z.open("INTL.txt") as fh:
        for raw in fh:
            line = raw.decode("utf-8")
            sid = line[line.find('"series_id":"') + 13:line.find('"', line.find('"series_id":"') + 13)]
            for pat, name, k in pats:
                m = pat.fullmatch(sid)
                if m:
                    for year, v in json.loads(line)["data"]:
                        if isinstance(v, (int, float)) and 2000 <= int(year) <= 2022:
                            rows.append((m.group(1), int(year), name, float(v) * k))
                    break
    long = pd.DataFrame(rows, columns=["iso3", "year", "col", "v"])
    wide = long.pivot_table(index=["iso3", "year"], columns="col", values="v").reset_index()
    return fill_tail(fill_tail(wide, "eia_bunker_twh", "eia_oil_twh"), "eia_flared_twh")


UN_SOLID = ["0100", "0200", "1100", "2000"]
UN_CRUDE = ["4100", "4200", "4300", "4400", "4500"]
UN_PRODUCTS = ["4610", "4620", "4630", "4640", "4651", "4652", "4653", "4661", "4669", "4670", "4680",
               "4691", "4692", "4693", "4694", "4695", "4699"]
UN_GAS = ["3000"]
UN_NAMES = {"Bonaire Sint Eustatius and Saba": "535", "Brunei": "096", "Cape Verde": "132", "Curacao": "531",
            "Czechia": "203", "Democratic Republic of Congo": "180", "East Timor": "626", "Ethiopia": "231",
            "Faroe Islands": "234", "France": "251", "Hong Kong": "344", "Iran": "364", "Italy": "382",
            "Laos": "418", "Macao": "446", "Micronesia (country)": "583", "Moldova": "498",
            "North Korea": "408", "Norway": "579", "Palestine": "275", "Russia": "643", "South Korea": "410",
            "Switzerland": "757", "Syria": "760", "Taiwan": "158", "Tanzania": "834", "Vietnam": "704",
            "Wallis and Futuna": "876"}


def read_unsd(countries):
    d = DATA / "external/unsd"
    x = (d / "dsd.xml").read_text(encoding="utf-8")
    i = x.find('id="CL_AREA_NRG"')
    block = x[i:x.find("</structure:Codelist>", i)]
    by_name = {n.lower(): c for c, n in re.findall(
        r'<structure:Code id="([^"]+)"[^>]*>.*?<common:Name xml:lang="en">([^<]+)</common:Name>', block, re.S)}
    m49 = {c: UN_NAMES.get(c, by_name.get(c.lower())) for c in countries}
    assert all(m49.values()), [c for c, v in m49.items() if not v]
    frames = []
    for code in UN_SOLID + UN_CRUDE + UN_PRODUCTS + UN_GAS:
        f = pd.read_csv(d / f"c_{code}.csv", dtype={"REF_AREA": str, "COMMODITY": str, "TRANSACTION": str})
        frames.append(f.assign(twh=f["OBS_VALUE"] * f["CONVERSION_FACTOR"] / 3600.0))
    u = pd.concat(frames)
    t = u.pivot_table(index=["REF_AREA", "TIME_PERIOD"], columns=["COMMODITY", "TRANSACTION"],
                      values="twh", aggfunc="sum").fillna(0.0)

    def ga(codes, tr="GA"):
        cols = [(c, tr) for c in codes if (c, tr) in t.columns]
        return t[cols].sum(axis=1)
    out = pd.DataFrame({
        "un_solid_twh": ga(UN_SOLID),
        "un_liquid_twh": ga(UN_PRODUCTS) + (ga(UN_CRUDE) - ga(UN_CRUDE, "086")).clip(lower=0.0),
        "un_gas_twh": ga(UN_GAS),
        "un_nonenergy_twh": ga(UN_SOLID + UN_CRUDE + UN_PRODUCTS + UN_GAS, "11"),
        "un_flared_twh": ga(UN_GAS, "104").where(ga(UN_GAS, "104") > 0, ga(UN_GAS, "104A")),
    }).reset_index().rename(columns={"REF_AREA": "m49", "TIME_PERIOD": "year"})
    back = {v: k for k, v in m49.items()}
    out["country"] = out["m49"].map(back)
    return out.dropna(subset=["country"]).drop(columns="m49")


def main():
    host = pd.concat([pd.read_csv(DATA / "train.csv", usecols=["country", "year"]),
                      pd.read_csv(DATA / "test.csv", usecols=["country", "year"])])
    owid = pd.read_csv(DATA / "external/owid/owid-energy-data.csv",
                       usecols=["country", "year", "iso_code", "biofuel_consumption"])
    iso = owid.dropna(subset=["iso_code"]).drop_duplicates("country").set_index("country")["iso_code"]
    iso = {**iso.to_dict(), **ISO_EXTRA}
    host["iso3"] = host["country"].map(iso)
    assert host["iso3"].notna().all(), host.loc[host["iso3"].isna(), "country"].unique()

    out = host.merge(owid[["country", "year", "biofuel_consumption"]], on=["country", "year"], how="left")
    for code, name in WDI.items():
        recs = json.loads((DATA / f"external/wdi/{code}.json").read_text())[1]
        df = pd.DataFrame({"iso3": [r["countryiso3code"] for r in recs],
                           "year": [int(r["date"]) for r in recs],
                           name: [r["value"] for r in recs]}).dropna()
        out = out.merge(df, on=["iso3", "year"], how="left")
    out = out.merge(read_eia(), on=["iso3", "year"], how="left")
    out = out.merge(read_unsd(host["country"].unique()), on=["country", "year"], how="left")
    out = out.drop(columns="iso3").sort_values(["country", "year"])
    out.to_csv(DATA / "ext_features.csv", index=False)
    print(f"wrote ext_features.csv: {out.shape}")
    print(out.drop(columns=["country", "year"]).notna().mean().round(3).to_string())


if __name__ == "__main__":
    main()
