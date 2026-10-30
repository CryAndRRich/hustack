from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "extra" / "probes"
LEADS = ["Q_pred_t", "Q_pred_t1", "Q_pred_t2", "Q_pred_t3"]


def read(name):
    return pd.read_csv(OUT / name)


def write(frame, name):
    frame.to_csv(OUT / name, index=False, float_format="%.6g")
    print(f"wrote extra/probes/{name}")


def masked(frame, mask):
    return frame.assign(**{c: frame[c] * mask for c in LEADS})


def leads(sub, s, basin, name):
    write(sub[["Unique_ID"]].assign(**{c: s.groupby(basin).shift(-k).groupby(basin).ffill()
                                       for k, c in enumerate(LEADS)}), name)


def make():
    sub = pd.read_csv(ROOT / "data/ExampleSubmission.csv")
    basin = sub.Unique_ID.str[-8:]
    v1, v21 = read("v1.csv"), read("v2.1.csv")
    assert (v1.Unique_ID == sub.Unique_ID).all() and (v21.Unique_ID == sub.Unique_ID).all()
    write(masked(v1, 0.0), "zero.csv")
    for b in sorted(basin.unique()):
        write(masked(v1, basin == b), f"v1_{b}.csv")
    write(masked(v21, basin == "08202700"), "v21_08202700.csv")


    nwm = pd.read_csv(ROOT / "data/nwm21.zip", dtype={"basin": str}, parse_dates=["date"])
    nwm = nwm[nwm.date.dt.year == 2018]
    nwm = sub.Unique_ID.map(dict(zip(nwm.date.dt.strftime("%Y-%m-%d %H:%M:%S") + "_" + nwm.basin, nwm.nwm21_mm_per_hour)))
    assert nwm.notna().all()
    leads(sub, nwm, basin, "nwm.csv")
    for b in ["08195000", "08196000", "08198500", "08200000"]:
        write(masked(read("nwm.csv"), basin == b), f"nwm_{b}.csv")
    b2 = read("best2.csv")
    leads(sub, b2.Q_pred_t.groupby(basin).transform(lambda x: x.rolling(13, center=True, min_periods=1).mean()), basin,
          "best2_smooth13.csv")
    leads(sub, b2.Q_pred_t.groupby(basin).shift(3).groupby(basin).bfill(), basin, "best2_late3.csv")
    write(b2.assign(**{c: 0.0 for c in LEADS[:3]}), "best2_col3.csv")


if __name__ == "__main__":
    make()
