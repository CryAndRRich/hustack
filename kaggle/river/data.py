import re
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from preprocess import FORCINGS, TARGET, find_data, read_competition

BASINS = ["08178880", "08195000", "08196000", "08198500", "08200000", "08202700"]
CUTOFF = pd.Timestamp("2017-01-01")
START = pd.Timestamp("1980-01-01")
END = pd.Timestamp("2019-01-01")
QCOL = "QObs(mm/h)"
GAUCH = "https://zenodo.org/records/4072701/files/"
NWM = "https://zenodo.org/records/16763144/files/NWM.7z"
LAGS = range(-6, 7)
TMP = Path("/tmp/ext")
OUT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")


def get(url, stream=False):
    for k in range(5):
        try:
            r = requests.get(url, stream=stream, timeout=120)
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            print(f"  retry {k + 1} {url}: {e}")
            time.sleep(30)
    raise RuntimeError(f"download failed: {url}")


def basin_of(name):
    hits = [b for b in BASINS if b in Path(name).name]
    return hits[0] if len(hits) == 1 else None


def read_gauch(url, kind):
    t0, out = time.time(), {}
    r = get(url, stream=True)
    r.raw.decode_content = True
    with tarfile.open(fileobj=r.raw, mode="r|gz") as tf:
        for m in tf:
            b = basin_of(m.name)
            if not (m.isfile() and m.name.endswith(".csv") and b):
                continue
            d = pd.read_csv(BytesIO(tf.extractfile(m).read()), parse_dates=["date"])
            assert b not in out, f"{kind}: two files for {b}"
            out[b] = d
            print(f"  {kind} {m.name}: {d.shape}, {d.date.min()} -> {d.date.max()}, columns {list(d.columns)}")
            if len(out) == len(BASINS):
                break
    print(f"  {kind}: {len(out)} basins ({time.time() - t0:.0f}s)")
    assert sorted(out) == BASINS, f"{kind}: found {sorted(out)}"
    return pd.concat([d.assign(basin=b) for b, d in out.items()], ignore_index=True)


def on_comp_rows(ext, comp, cols, lag=0):
    k = comp[["basin"]].assign(date=comp.date + pd.Timedelta(hours=lag))
    return ext.set_index(["basin", "date"])[cols].reindex(pd.MultiIndex.from_frame(k)).to_numpy()


def corr(a, b):
    ok = ~(np.isnan(a) | np.isnan(b))
    return np.corrcoef(a[ok], b[ok])[0, 1]


def align(ext, comp, cols, label):
    print(f"== {label}")
    a = comp[cols].to_numpy(float)
    by_lag = {lag: on_comp_rows(ext, comp, cols, lag) for lag in LAGS}
    for j, c in enumerate(cols):
        r = {lag: corr(a[:, j], v[:, j]) for lag, v in by_lag.items()}
        lag = max(r, key=r.get)
        assert lag == 0, f"{c}: best correlation at lag {lag} h ({r[lag]:.4f}, lag 0 {r[0]:.4f}): time zones differ"
        x, y = by_lag[0][:, j], a[:, j]
        ok = ~(np.isnan(x) | np.isnan(y))
        assert ok.mean() > 0.5, f"{c}: only {ok.mean():.3f} of competition rows matched"
        tol = dict(rtol=1e-3, atol=1e-5 * np.abs(y[ok]).max())
        if not np.allclose(x[ok], y[ok], **tol):
            slope, icpt = np.polyfit(x[ok], y[ok], 1)
            print(f"  {c}: raw max |diff| {np.abs(x[ok] - y[ok]).max():.3g}; mapping ext * {slope:.6g} + {icpt:.6g}")
            x = slope * x + icpt
            ext[c] = slope * ext[c] + icpt
            assert np.allclose(x[ok], y[ok], **tol), f"{c}: no linear map matches (max |diff| {np.abs(x[ok] - y[ok]).max():.3g})"
        print(f"  {c}: matched {ok.mean():.4f} of rows, lag 0 (corr {r[0]:.6f}), max |diff| {np.abs(x[ok] - y[ok]).max():.3g}")
    return ext


def unpack_7z(path, targets):
    if shutil.which("7z"):
        subprocess.run(["7z", "x", "-y", f"-o{TMP}", str(path)] + targets, check=True, stdout=subprocess.DEVNULL)
        return
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "py7zr"], check=True)
    import py7zr
    with py7zr.SevenZipFile(path) as z:
        z.extract(path=TMP, targets=targets)


def nwm_v21(ds, name):
    timed = [v for v in ds.data_vars if ds[v].ndim >= 1 and np.issubdtype(ds[ds[v].dims[0]].dtype, np.datetime64)]
    for v in timed:
        print(f"    {name} var {v}: dims {ds[v].dims}, attrs {dict(ds[v].attrs)}")
    tag = {v: (v + " " + " ".join(map(str, ds[v].attrs.values()))).lower() for v in timed}
    v21 = [v for v in timed if re.search(r"(v|nwm|_)2[._]?1\b|2\.1", tag[v])]
    if not v21 and len(timed) == 1 and re.search(r"2\.1|v2[._]?1", str(ds.attrs).lower()):
        v21 = timed
    assert len(v21) == 1, f"{name}: cannot tell the v2.1 series apart; time variables {timed}, file attrs {ds.attrs}"
    s = ds[v21[0]].squeeze().to_series()
    idx = pd.DatetimeIndex(s.index)
    s.index = (idx.tz_convert(None) if idx.tz is not None else idx).floor("h")
    return s.astype(float)


def probe_nwm(q, area):
    TMP.mkdir(parents=True, exist_ok=True)
    path = TMP / "NWM.7z"
    with open(path, "wb") as f:
        for chunk in get(NWM, stream=True).iter_content(1 << 24):
            f.write(chunk)
    names = [f"NWM/NWM_{b}.nc" for b in BASINS]
    unpack_7z(path, names)
    path.unlink()
    import xarray as xr
    parts = []
    for n in names:
        b = basin_of(n)
        ds = xr.open_dataset(TMP / n)
        print(f"  {n}:\n{ds}")
        s = nwm_v21(ds, n)
        parts.append(pd.DataFrame({"basin": b, "date": s.index, "nwm21_mm_per_hour": s.values * 3.6 / area[b]}))
    nwm = pd.concat(parts, ignore_index=True).drop_duplicates(["basin", "date"])
    nwm = nwm[(nwm.date >= START) & (nwm.date < END)]
    print(nwm.groupby("basin").date.agg(["min", "max", "size"]).to_string())
    m = q.merge(nwm, on=["basin", "date"]).dropna()
    for b, g in [("all", m)] + list(m.groupby("basin")):
        o, s = g[TARGET].values, g.nwm21_mm_per_hour.values
        print(f"  NWM v2.1 vs observed 1980-2016, {b} ({len(g)} h): rmse {np.sqrt(np.mean((o - s) ** 2)):.5f}, "
              f"zero {np.sqrt(np.mean(o ** 2)):.5f}, NSE {1 - np.sum((o - s) ** 2) / np.sum((o - o.mean()) ** 2):.3f}, "
              f"corr {np.corrcoef(o, s)[0, 1]:.3f}, volume ratio {s.sum() / o.sum():.3f}")
    csv = OUT / "ext_nwm21.csv"
    nwm.assign(date=nwm.date.dt.strftime("%Y-%m-%d %H:%M:%S")).to_csv(csv, index=False)
    with zipfile.ZipFile(OUT / "nwm21.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(csv, csv.name)
    csv.unlink()
    print(f"DATASET: nwm21.zip ({(OUT / 'nwm21.zip').stat().st_size / 1e6:.0f} MB)")


def build_camels(df):
    q = read_gauch(GAUCH + "usgs_streamflow_csv.tar.gz", "streamflow")
    assert QCOL in q.columns, f"{QCOL} missing: {list(q.columns)}"
    q = q.rename(columns={QCOL: TARGET})[["basin", "date", TARGET]].drop_duplicates(["basin", "date"])
    q.loc[q[TARGET] < 0, TARGET] = np.nan
    q = q[(q.date >= START) & (q.date < CUTOFF)].reset_index(drop=True)
    align(q, df[df.part == "train"][["basin", "date", TARGET]], [TARGET], "Gauch discharge vs competition train")
    print("  observed share of hours per year:")
    full = pd.MultiIndex.from_product([BASINS, pd.date_range(START, CUTOFF, freq="h", inclusive="left")],
                                      names=["basin", "date"])
    cov = q.set_index(["basin", "date"])[TARGET].reindex(full).notna()
    print(cov.groupby([cov.index.get_level_values(1).year, cov.index.get_level_values(0)]).mean()
          .unstack().round(2).to_string())

    f = read_gauch(GAUCH + "nldas_hourly_csv.tar.gz", "nldas")
    missing = [c for c in FORCINGS if c not in f.columns]
    assert not missing, f"NLDAS columns missing: {missing}; got {list(f.columns)}"
    f = f[["basin", "date"] + FORCINGS].drop_duplicates(["basin", "date"])
    f = f[(f.date >= START) & (f.date < END)].reset_index(drop=True)
    f = align(f, df[["basin", "date"] + FORCINGS], FORCINGS, "Gauch NLDAS vs competition forcings (train + test)")
    hours = f.groupby("basin").date.agg(["min", "max", "size"])
    print(hours.to_string())
    print(f"  NLDAS hours contiguous: {(hours['size'] == (hours['max'] - hours['min']) / pd.Timedelta('1h') + 1).all()}")

    ext = f.merge(q, on=["basin", "date"], how="left")
    assert ext.loc[ext.date >= CUTOFF, TARGET].isna().all()
    print(f"ext_hourly: {ext.shape}; observed q per basin: {ext.groupby('basin')[TARGET].count().to_dict()}")
    csv = OUT / "ext_hourly.csv"
    ext.assign(date=ext.date.dt.strftime("%Y-%m-%d %H:%M:%S")).to_csv(csv, index=False)
    with zipfile.ZipFile(OUT / "camelshourly.zip", "w", zipfile.ZIP_DEFLATED) as z:
        z.write(csv, csv.name)
    csv.unlink()
    print(f"DATASET: camelshourly.zip ({(OUT / 'camelshourly.zip').stat().st_size / 1e6:.0f} MB)")
    return q


def run():
    paths = find_data()
    df = read_competition(paths)
    area = df.groupby("basin").area_gages2.first()
    p = paths["ext_hourly.csv"]
    if not p.exists():
        q = build_camels(df)
    else:
        print(f"camelshourly found ({p}): not rebuilt")
        q = pd.read_csv(p, usecols=["basin", "date", TARGET], dtype={"basin": str}, parse_dates=["date"]).dropna()
    probe_nwm(q, area)


if __name__ == "__main__":
    run()
