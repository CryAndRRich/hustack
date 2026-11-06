import subprocess
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data"
SRC_DIR = DATA / "external/moosavi2020_diversity/data_paper/features_labels"
SRC_TARGET = "pure_uptake_CO2_298.00_15000"
TARGET = "CO2 uptake at 0.15 bar and 298K"

if __name__ == "__main__":
    tr, te, ex = pd.read_csv(DATA / "train.csv"), pd.read_csv(DATA / "test.csv"), pd.read_csv(SRC_DIR / "CoRE2019_alldata.csv")
    feats = [c for c in te.columns if c != "MOFname"]
    m = tr.merge(ex[["MOFname", SRC_TARGET, *feats]], on="MOFname", suffixes=("", "_e"))
    assert len(m) == len(tr) and (m[TARGET] - m[SRC_TARGET]).abs().max() == 0
    assert max((m[c] - m[f"{c}_e"]).abs().max() for c in feats) < 1e-6
    x = ex[~ex.MOFname.isin(tr.MOFname) & ~ex.MOFname.isin(te.MOFname)]
    out = x[["MOFname", *feats, SRC_TARGET]].rename(columns={SRC_TARGET: TARGET})
    assert out.MOFname.is_unique and out.notna().all().all()
    out.to_csv(DATA / "ext_core2019.csv", index=False)
    hy = pd.concat([pd.read_csv(SRC_DIR / f, usecols=["MOFname", *feats, SRC_TARGET]) for f in ("BW20K_alldata.csv", "ARABG_alldata.csv")])
    hy = hy[["MOFname", *feats, SRC_TARGET]].rename(columns={SRC_TARGET: TARGET})
    assert hy.MOFname.is_unique and hy.notna().all().all() and not hy.MOFname.isin(ex.MOFname).any()
    hy.to_csv(DATA / "ext_hypo.csv", index=False)
    files = [str(DATA / f) for f in ("ext_core2019.csv", "ext_hypo.csv")]
    subprocess.run(["zip", "-j", "-q", "-o", str(DATA / "core2019-extra.zip"), *files], check=True)
    print(f"wrote ext_core2019.csv: {len(out)} rows, ext_hypo.csv: {len(hy)} rows, {len(feats)} features; zipped to core2019-extra.zip")
