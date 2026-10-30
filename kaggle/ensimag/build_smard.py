\
\
\
\
\
\
\

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "external" / "smard"
TZ = "Europe/Berlin"
BASE = "https://www.smard.de/app/chart_data"
PAUSE_S = 0.6
SERIES = {"fc_onshore": 123, "fc_offshore": 3791, "fc_pv": 125, "fc_load": 411}


def fetch(url, path):
    if path.exists() and path.stat().st_size > 0:
        return json.loads(path.read_text())
    tmp = path.with_suffix(".part")
    r = subprocess.run(["curl", "-sf", "--max-time", "60", "-o", str(tmp), url])
    if r.returncode != 0:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"download failed (curl exit {r.returncode}): {url} — stop and fetch it by hand")
    tmp.rename(path)
    time.sleep(PAUSE_S)
    return json.loads(path.read_text())


def load_series(name, start, end):
    fid = SERIES[name]
    idx = fetch(f"{BASE}/{fid}/DE/index_quarterhour.json", RAW / f"index_{fid}_DE_quarterhour.json")["timestamps"]
    lo = (pd.Timestamp(start, tz=TZ) - pd.Timedelta(days=7)).value // 10**6
    hi = pd.Timestamp(end, tz=TZ).value // 10**6
    chunks = [t for t in idx if lo <= t <= hi]
    assert chunks, f"{name}: no SMARD chunk between {start} and {end} — the index may be stale, delete {RAW}/index_*"
    parts = []
    for i, t in enumerate(chunks):
        js = fetch(f"{BASE}/{fid}/DE/{fid}_DE_quarterhour_{t}.json", RAW / f"{fid}_DE_quarterhour_{t}.json")
        parts.append(pd.DataFrame(js["series"], columns=["ms", "v"]))
        if (i + 1) % 50 == 0:
            print(f"   {name}: {i + 1}/{len(chunks)} chunks")
    s = pd.concat(parts).dropna().drop_duplicates("ms")
    out = pd.Series(s["v"].astype(float).values * 4.0, index=pd.to_datetime(s["ms"], unit="ms"), name=name).sort_index()
    print(f"{name}: {len(chunks)} chunks, {len(out)} values {out.index.min()} -> {out.index.max()}")
    return out


def build(start, end, out_dir):
    RAW.mkdir(parents=True, exist_ok=True)
    grid = pd.date_range(pd.Timestamp(start, tz=TZ).tz_convert("UTC").tz_localize(None),
                         pd.Timestamp(end, tz=TZ).tz_convert("UTC").tz_localize(None), freq="15min", inclusive="left")
    q = pd.DataFrame({name: load_series(name, start, end).reindex(grid) for name in SERIES}, index=grid)
    q.index.name = "date"

    h = pd.read_csv(ROOT / "data" / "competition" / "train.csv", usecols=["date", "load"], parse_dates=["date"])
    j = h.set_index("date").join(q["fc_load"], how="inner").dropna()
    exact = (np.abs(j["load"] - j["fc_load"]) < 0.5).mean()
    print(f"alignment: host load == SMARD load forecast x4 on {exact:.4f} of {len(j)} shared train slots")
    assert exact > 0.99, "SMARD timestamps do not line up with the host's UTC grid"

    qh = pd.DataFrame({"smard_wind_fc": q["fc_onshore"] + q["fc_offshore"], "smard_offshore_fc": q["fc_offshore"],
                       "smard_pv_fc": q["fc_pv"]}, index=q.index)
    out_dir.mkdir(parents=True, exist_ok=True)
    qh.round(2).to_csv(out_dir / "ext_smard_qh.csv")
    print(f"wrote {out_dir / 'ext_smard_qh.csv'} {qh.shape}, NaN per column {qh.isna().sum().to_dict()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2019-12-01")
    ap.add_argument("--end", default="2024-09-10")
    ap.add_argument("--out", default=str(ROOT / "data"))
    a = ap.parse_args()
    build(a.start, a.end, Path(a.out))
