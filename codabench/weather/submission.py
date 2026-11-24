import csv
import math
import zipfile
from datetime import datetime, timedelta

import numpy as np

from config import CASES, CORRIDORS, DEPARTURES, DTFMT, MAX_HS, MAX_WIND, NM, TEAM, WP_PER_H
from land import on_land

FILE_A_COLUMNS = ["departure_time_utc", "arrival_time_utc", "energy_cons_mwh", "max_wind_mps", "max_hs_m",
                  "sailed_distance_nm", "details_filename"]


def corridor_of(case):
    return "atlantic" if case[0] == "A" else "pacific"


def track_name(sub_no, case, d):
    return f"{TEAM}-{sub_no}-{case}-{DEPARTURES[d]:%Y%m%d}.csv"


def sailed_nm(lat, lon):
    la1, la2 = np.radians(lat[:-1]), np.radians(lat[1:])
    dlat, dlon = la2 - la1, np.radians(lon[1:] - lon[:-1])
    a = np.sin(dlat / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin(dlon / 2) ** 2
    return float(np.sum(6371000.0 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))) / NM


def write_submission(root, tracks, report, sub_no, deps):
    (root / "tracks").mkdir(parents=True, exist_ok=True)
    for case in CASES:
        passage = CORRIDORS[corridor_of(case)]["passage_h"]
        rows = []
        for d in deps:
            dep = DEPARTURES[d]
            lat, lon = tracks[(case, d)]
            assert len(lat) == passage * WP_PER_H + 1
            times = np.datetime64(dep) + np.arange(len(lat)) * np.timedelta64(60 // WP_PER_H, "m")
            ts = np.char.replace(np.datetime_as_string(times, unit="s"), "T", " ")
            body = np.char.add(np.char.add(np.char.add(ts, ","), np.char.mod("%.6f", lat)),
                               np.char.add(",", np.char.mod("%.6f", lon)))
            name = track_name(sub_no, case, d)
            (root / "tracks" / name).write_text("time_utc,lat_deg,lon_deg\n" + "\n".join(body) + "\n")
            e, mw, mh = report[(case, d)]
            rows.append([dep.strftime(DTFMT), (dep + timedelta(hours=passage)).strftime(DTFMT), f"{e:.6f}",
                         f"{mw:.4f}", f"{mh:.4f}", f"{sailed_nm(lat, lon):.4f}", name])
        with (root / f"{TEAM}-{sub_no}-{case}.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(FILE_A_COLUMNS)
            w.writerows(rows)


def validate(root, sub_no, deps, union):
    errors = []
    for case in CASES:
        c = CORRIDORS[corridor_of(case)]
        gc = "GC" in case
        path = root / f"{TEAM}-{sub_no}-{case}.csv"
        with path.open() as f:
            rows = list(csv.DictReader(f))
        if len(rows) != len(deps):
            errors.append(f"{path.name}: {len(rows)} rows")
        for r, d in zip(rows, deps):
            dep, arr = datetime.strptime(r["departure_time_utc"], DTFMT), datetime.strptime(r["arrival_time_utc"], DTFMT)
            if dep != DEPARTURES[d] or abs((arr - dep).total_seconds() / 3600 - c["passage_h"]) > 1.0:
                errors.append(f"{path.name}: schedule/passage {r['departure_time_utc']}")
            vals = [float(r[k]) for k in ("energy_cons_mwh", "max_wind_mps", "max_hs_m", "sailed_distance_nm")]
            if any(v != v or v < 0 for v in vals):
                errors.append(f"{path.name}: bad numeric {vals}")
            if not gc and (vals[1] > MAX_WIND or vals[2] > MAX_HS):
                errors.append(f"{path.name}: limit {r['departure_time_utc']} wind {vals[1]} hs {vals[2]}")
            with (root / "tracks" / r["details_filename"]).open() as f:
                pts = list(csv.DictReader(f))
            t = [datetime.strptime(p["time_utc"], DTFMT) for p in pts]
            lat = np.array([float(p["lat_deg"]) for p in pts])
            lon = np.array([float(p["lon_deg"]) for p in pts])
            if len(pts) < 2 or any(b <= a for a, b in zip(t[:-1], t[1:])):
                errors.append(f"{r['details_filename']}: times")
            if t[0] != dep or t[-1] != arr:
                errors.append(f"{r['details_filename']}: endpoint times")
            for (la, lo), (ela, elo) in (((lat[0], lon[0]), c["src"]), ((lat[-1], lon[-1]), c["dst"])):
                if math.hypot(la - ela, (lo - elo + 180) % 360 - 180) > 0.5:
                    errors.append(f"{r['details_filename']}: endpoint position")
            if np.any(np.abs(lat) > 90) or np.any(np.abs(lon) > 360):
                errors.append(f"{r['details_filename']}: bounds")
            if not gc:
                n_land = int(on_land(union, lat, lon).sum())
                if n_land:
                    errors.append(f"{r['details_filename']}: {n_land} points on land")
    return errors


def make_zip(root, zip_path):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(root.rglob("*.csv")):
            zf.write(p, p.relative_to(root).as_posix())
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
    n_a = sum("/" not in n for n in names)
    print(f"{zip_path.name}: {len(names)} files ({n_a} at root), {zip_path.stat().st_size / 1e6:.1f} MB", flush=True)
    return n_a
