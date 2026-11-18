import gzip
import json
import subprocess
import time
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
RAW = DATA / "external"
WFS = ("https://portal.csdi.gov.hk/server/services/common/landsd_rcd_1637211194312_35158/MapServer/WFSServer"
       "?service=wfs&version=2.0.0&request=GetFeature&typeNames=csdi:Building&outputFormat=GEOJSON"
       "&count={n}&startIndex={i}&bbox={b},urn:ogc:def:crs:EPSG::4326")
BBOX = (22.285, 114.155, 22.345, 114.235)
PAGE = 5000
BRDC = "https://igs.bkg.bund.de/root_ftp/IGS/BRDC/2021/{d}/BRDM00DLR_S_2021{d}0000_01D_MN.rnx.gz"
SYSTEMS = "GREC"
DOYS = ("137", "138", "141", "337")


def get(url, path):
    if not path.exists():
        subprocess.run(["curl", "-sfL", "-m", "300", "-A", "generic-client", "-o", str(path), url], check=True)
    return path


def buildings():
    out = RAW / "landsd_building"
    out.mkdir(parents=True, exist_ok=True)
    rows, i = [], 0
    while True:
        url = WFS.format(n=PAGE, i=i, b=",".join(map(str, BBOX)))
        feats = json.loads(get(url, out / f"page_{i}.json").read_text())["features"]
        for f in feats:
            g, p = f["geometry"], f["properties"]
            polys = g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]
            rings = "|".join(";".join(f"{x:.7f} {y:.7f}" for x, y in ring) for poly in polys for ring in poly)
            rows.append((p["BuildingCSUID"], p["BuildingBlockType"], p["Status"], p["DateCreate"],
                         p["BaseHeight"], p["TopHeight"], rings))
        print(f"page {i}: {len(feats)} features")
        if len(feats) < PAGE:
            break
        i += PAGE
        time.sleep(1)
    df = pd.DataFrame(rows, columns=["uid", "block_type", "status", "date_create", "base_mpd", "top_mpd", "rings"])
    df = df.drop_duplicates("uid")
    df.to_csv(DATA / "ext_buildings.csv", index=False)
    print(f"ext_buildings.csv: {len(df)} buildings, top height present {df.top_mpd.notna().mean():.3f}")


def ephemeris():
    out = RAW / "brdc"
    out.mkdir(parents=True, exist_ok=True)
    for d in DOYS:
        gz = get(BRDC.format(d=d), out / f"BRDM_2021{d}.rnx.gz")
        lines = gzip.open(gz, "rt").read().splitlines(keepends=True)
        end = next(i for i, x in enumerate(lines) if "END OF HEADER" in x) + 1
        keep, on = lines[:end], False
        for x in lines[end:]:
            if not x.startswith(" "):
                on = x[0] in SYSTEMS
            if on:
                keep.append(x)
        (DATA / f"brdc_2021{d}.rnx").write_text("".join(keep))
        print(f"brdc_2021{d}.rnx: {(DATA / f'brdc_2021{d}.rnx').stat().st_size / 1e6:.1f} MB")


def host_tables():
    cols = ["GPS_Time(s)", "CNR", "Elevation", "Azimuth", "Pr_Residual", "Pr_residual_Root_of_sum_of_square_error",
            "Pr_Residual_cigma"]
    for part in ("train", "test"):
        x = pd.read_excel(DATA / "raw" / f"GNSS_{part}.xlsx")[cols]
        x.to_csv(DATA / f"ext_host_{part}.csv", index=False)
        print(f"ext_host_{part}.csv: {len(x)} rows")


if __name__ == "__main__":
    host_tables()
    ephemeris()
    buildings()
