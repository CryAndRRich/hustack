import os
import struct
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

TARGETS = ["t_ch4", "t_no2", "t_co", "t_aai", "t_firms"]
FIRE_FLOOR = -0.439
BANDS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]
CLOUDY = [0, 1, 3, 8, 9, 10]
CNN_SIZE = 128
FM_SIZE = 128


def find_data_dir():
    for root in ("/kaggle/input", str(Path.cwd() / "data")):
        for path, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if d != "images"]
            if "train.csv" in files and "test.csv" in files and os.path.isdir(os.path.join(path, "images")):
                return Path(path)
    try:
        return Path(__file__).resolve().parent.parent / "data"
    except NameError:
        return Path.cwd() / "data"


def work_dir():
    return Path("/kaggle/working") if os.path.isdir("/kaggle/working") else Path(".")


def load_tables(data_dir):
    tr = pd.read_csv(data_dir / "train.csv", dtype={"sample_id": str, "site_key": str})
    te = pd.read_csv(data_dir / "test.csv", dtype={"sample_id": str})
    ss = pd.read_csv(data_dir / "sample_submission.csv", dtype={"sample_id": str})
    assert "site_key" in tr and list(ss.sample_id) == list(te.sample_id)
    return tr, te, ss


def read_tif(path):
    b = Path(path).read_bytes()
    e = "<" if b[:2] == b"II" else ">"
    off = struct.unpack(e + "I", b[4:8])[0]
    tags = {}
    for i in range(struct.unpack(e + "H", b[off:off + 2])[0]):
        tag, typ, cnt, val = struct.unpack(e + "HHII", b[off + 2 + 12 * i:off + 14 + 12 * i])
        if cnt == 1:
            tags[tag] = val & 0xFFFF if typ == 3 else val
        else:
            fmt = {3: "H", 4: "I"}[typ]
            tags[tag] = struct.unpack(e + fmt * cnt, b[val:val + cnt * struct.calcsize(fmt)])
    assert tags[259] == 8 and tags[284] == 1 and tags[278] == 1 and tags[277] == 14, f"unexpected TIFF layout {path}"
    w, s, dt = tags[256], tags[277], np.dtype(e + "u2")
    a = np.stack([np.frombuffer(zlib.decompress(b[o:o + c]), dt).reshape(w, s) for o, c in zip(tags[273], tags[279])])
    if tags.get(317, 1) == 2:
        a = np.cumsum(a, axis=1, dtype=np.uint16)
    return a.transpose(2, 0, 1)


def chip_stats(a):
    x = a[:12].astype(np.float32)
    scl, cld = a[12], a[13]
    h, w = scl.shape
    valid = x[3] > 0
    nocloud = valid & ~np.isin(scl, CLOUDY)
    clear = nocloud & (cld < 50)
    if clear.sum() < 500:
        clear = valid if valid.sum() >= 500 else np.ones_like(valid)
    r = {"clear_frac": clear.mean(), "cld_mean": cld.mean(), "zero_frac": (x == 0).mean()}
    r.update({f"scl{c}": v for c, v in enumerate(np.bincount(scl.ravel(), minlength=12)[:12] / scl.size)})
    q = np.percentile(x[:, clear], [5, 50, 95, 99.9], axis=1)
    for k, b in enumerate(BANDS):
        r.update({f"{b}_p5": q[0, k], f"{b}_p50": q[1, k], f"{b}_p95": q[2, k], f"{b}_p999": q[3, k]})
    yy, xx = np.ogrid[:h, :w]
    centre = clear & ((yy - h / 2) ** 2 + (xx - w / 2) ** 2 <= (min(h, w) / 6) ** 2)
    for b in ("B4", "B8", "B11", "B12"):
        k = BANDS.index(b)
        r[f"{b}_ctr"] = np.median(x[k][centre]) - q[1, k] if centre.sum() >= 100 else 0.0
    b8a, b11, b12 = x[8], x[10], x[11]
    cand = nocloud & (b12 > 1500)
    tai = (b12 - b11) / np.maximum(b8a, 1.0)
    hot = cand & (tai >= 0.45) & (b12 - b11 >= b11 - b8a)
    r.update({"hot_n": hot.sum(), "b12_hi_n": (nocloud & (b12 > 6000)).sum(),
              "tai_max": tai[cand].max() if cand.any() else 0.0,
              "tai_p999": np.percentile(tai[cand], 99.9) if cand.any() else 0.0})
    return r


def cnn_input(a):
    x = a[:12].astype(np.float32) / 10000.0
    nocloud = ((x[3] > 0) & ~np.isin(a[12], CLOUDY)).astype(np.float32)
    b8a, b11, b12 = x[8], x[10], x[11]
    tai = np.where(b12 > 0.15, np.clip((b12 - b11) / np.maximum(b8a, 1e-4), -1, 3), -1).astype(np.float32)
    avg = F.adaptive_avg_pool2d(torch.from_numpy(np.stack([x[3], x[7], b8a, b11, b12, nocloud]))[None], CNN_SIZE)[0]
    mx = F.adaptive_max_pool2d(torch.from_numpy(np.stack([b12, tai]))[None], CNN_SIZE)[0]
    return torch.cat([avg, mx]).clamp(-1, 3).half()


def fm_input(a):
    x = F.adaptive_avg_pool2d(torch.from_numpy(a[:12].astype(np.float32))[None], FM_SIZE)[0]
    return x.round().clamp(0, 65535).numpy().astype(np.uint16)


class Chips(torch.utils.data.Dataset):
    def __init__(self, image_dir, ids):
        self.image_dir, self.ids = image_dir, ids

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        a = read_tif(self.image_dir / f"{self.ids[i]}.tif")
        return cnn_input(a), fm_input(a), chip_stats(a)


def collate(batch):
    return torch.stack([b[0] for b in batch]), np.stack([b[1] for b in batch]), [b[2] for b in batch]


def extract_chips(image_dir, ids, workers=4):
    t0 = time.time()
    loader = torch.utils.data.DataLoader(Chips(image_dir, ids), batch_size=32, num_workers=workers, collate_fn=collate)
    stats, imgs, fms = [], [], []
    for n, (x, fm, s) in enumerate(loader):
        stats += s
        imgs.append(x.numpy())
        fms.append(fm)
        if n % 40 == 0:
            print(f"  chips {len(stats)}/{len(ids)} | {time.time() - t0:.0f}s")
    stats = pd.DataFrame(stats, index=ids).astype(np.float32)
    imgs, fms = np.concatenate(imgs), np.concatenate(fms)
    print(f"chip pass {time.time() - t0:.0f}s, {stats.shape[1]} stats, cnn input {imgs.shape} {imgs.dtype}, "
          f"fm input {fms.shape} {fms.dtype}")
    return stats, imgs, fms


def tab_features(df):
    out = df.drop(columns=[c for c in ["sample_id", "site_key"] + TARGETS if c in df]).copy()
    th = np.deg2rad(df["wind_direction"])
    out["wind_u"] = -df["wind_speed"] * np.sin(th)
    out["wind_v"] = -df["wind_speed"] * np.cos(th)
    out["inv_wind"] = 1.0 / np.maximum(df["wind_speed"], 0.5)
    return out


def build_features(tr, te, stats):
    raw = pd.concat([tr, te], ignore_index=True)
    chip = stats.loc[raw.sample_id].reset_index(drop=True).add_prefix("c_")
    X = pd.concat([tab_features(raw).reset_index(drop=True), chip], axis=1)
    assert len(X) == len(raw)
    print(f"features: {X.shape[1]}")
    return X


EXT_PARAMS = dict(n_estimators=600, learning_rate=0.03, num_leaves=15, min_child_samples=30, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.6, reg_lambda=5.0, verbose=-1, n_jobs=4)


def load_ext(data_dir):
    cands = []
    for p, dirs, files in os.walk("/kaggle/input"):
        dirs[:] = [d for d in dirs if d != "images"]
        if "ext_rows.csv" in files:
            cands.append(Path(p) / "ext_rows.csv")
    cands.append(data_dir / "external" / "ext_rows.csv")
    for p in cands:
        if p.exists():
            e = pd.read_csv(p)
            print(f"external rows: {len(e)} from {e.fac_id.nunique()} facilities ({p})")
            return e
    print("external rows not found: arm E skipped")
    return None


def ext_tables(e):
    q = lambda x, s: np.round(x / s) * s
    t, td = e.temperature_2m - 273.15, e.dewpoint_temperature_2m - 273.15
    date = pd.to_datetime(e.date)
    m = date.dt.month
    f = pd.DataFrame({"month_sin": np.sin(2 * np.pi * m / 12).round(3), "month_cos": np.cos(2 * np.pi * m / 12).round(3),
                      "temperature": q(t, 0.5),
                      "humidity": q(100 * np.exp(17.625 * td / (243.04 + td) - 17.625 * t / (243.04 + t)), 2),
                      "wind_speed": q(e.ws, 0.5),
                      "wind_direction": q((180 + np.degrees(np.arctan2(e.u_component_of_wind_10m, e.v_component_of_wind_10m))) % 360, 10) % 360,
                      "precipitation": q(e.total_precipitation_sum, 0.0005),
                      "solar_rad": q(e.surface_solar_radiation_downwards_sum / 1e6, 0.5),
                      "elevation": q(e.elevation, 50), "slope": q(e.slope, 1), "s2_cloud_frac": e.s2_cloud_frac.round(3)})
    f = tab_features(f)
    chip = e[[c for c in e.columns if c.split("_")[0] in BANDS or c.startswith(("scl", "clear_frac", "cld_mean", "zero_frac",
                                                                                  "hot_n", "b12_hi_n", "tai_"))]]
    chip = chip.assign(**{f"{b}_ctr": e[f"{b}_ctr"] - e[f"{b}_p50"] for b in ("B4", "B8", "B11", "B12") if f"{b}_ctr" in e})
    f = pd.concat([f, chip.add_prefix("c_")], axis=1)
    y = pd.DataFrame({g: np.where(e[f"n_{g}_site"] > 0, e[f"y_{g}_site"] - e[f"y_{g}_ring"], np.nan) for g in ("ch4", "no2", "co")})
    aai = pd.Series(np.where(e.n_aai > 0, e.y_aai, np.nan), index=e.index)
    resid = aai - aai.groupby(e.fac_id).transform("mean")
    resid = resid - resid.groupby(m).transform("mean")
    quarter = date.dt.year * 4 + (m - 1) // 3
    y["aai"] = aai - resid.groupby(quarter).transform("mean").fillna(0.0)
    t21 = e["y_firms_T21"] if "y_firms_T21" in e else pd.Series(np.nan, index=e.index)
    y["fire"] = t21.notna().astype(float)
    y["firms"] = np.log(np.maximum(t21 - 300.0, 1.0))
    return f, y


def ext_features(Fdf, e, n_estimators):
    import lightgbm as lgb
    from sklearn.model_selection import GroupKFold
    t0 = time.time()
    f, y = ext_tables(e)
    cols = [c for c in Fdf.columns if c in f.columns]
    Xe, Xc = f[cols].to_numpy(np.float32), Fdf[cols].to_numpy(np.float32)
    params = dict(EXT_PARAMS, n_estimators=n_estimators)
    feats, groups = {}, e.fac_id.to_numpy()
    pool = {"X": Xe, "cols": cols, "y": {}, "oof": {}}
    for name in ("ch4", "no2", "co", "aai", "fire", "firms"):
        yy = y[name].to_numpy(float)
        keep = ~np.isnan(yy) & (y.fire.to_numpy() > 0 if name == "firms" else True)
        if keep.sum() < 200:
            print(f"  ext {name}: {keep.sum()} rows, skipped")
            continue
        lo, hi = np.nanpercentile(yy[keep], [0.5, 99.5])
        yk = np.clip(yy[keep], lo, hi)
        model = lgb.LGBMClassifier if name == "fire" else lgb.LGBMRegressor
        cv = np.zeros(keep.sum())
        for a, b in GroupKFold(3).split(Xe[keep], groups=groups[keep]):
            m = model(**params).fit(Xe[keep][a], yk[a])
            cv[b] = m.predict_proba(Xe[keep][b])[:, 1] if name == "fire" else m.predict(Xe[keep][b])
        m = model(**params).fit(Xe[keep], yk)
        feats[f"ext_{name}"] = m.predict_proba(Xc)[:, 1] if name == "fire" else m.predict(Xc)
        pool["y"][name], pool["oof"][name] = np.full(len(yy), np.nan), np.full(len(yy), np.nan)
        pool["y"][name][keep], pool["oof"][name][keep] = yk, cv
        print(f"  ext {name}: {keep.sum()} rows, facility-grouped 3-fold corr {np.corrcoef(cv, yk)[0, 1]:.3f}")
    print(f"external models: {len(cols)} shared features, {time.time() - t0:.0f}s")
    return pd.DataFrame(feats, index=Fdf.index), pool
