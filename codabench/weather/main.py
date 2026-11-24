import csv
import json
import math
import os
import pickle
import shutil
import subprocess
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy.ndimage as ndi
import torch

from config import (BANDS, BAND_SPAN, BATCH_DAYS, BOARD_BASELINE, BUDGET_H, CASES, CORRIDORS, DEPARTURES, DTFMT,
                    EARTH_R, GC_DP, MAX_HS, MAX_WIND, NEWTON, NEWTON_CHUNK, NEWTON_FINE, NM, O_DP, OPT_VARIANT, OUT,
                    RESERVE_H, SELECT, SMOKE, SUBMISSION_NO, TEAM, TUBE, TUBE2, V2_SUB, VARIANTS, WORK, WP_PER_H)
from dp import Lattice, _dev, _power_kernel, reference_check
from era5 import PAD, Corridor, Window, find_era5_dir, read_pack, spline_coef
from geo import Frame, latlon, leg_geometry, unit_vec
from land import LandRaster, land_union, on_land, read_polygons
from metric import baseline_points, bearing, dep_hour, evaluate, haversine, scorer_resample
from polish import newton, newton_selftest
from rise import power_angles, power_vec
from submission import make_zip, track_name, validate, write_submission

T_START = float(os.environ.get("ROUTING_T0", time.time()))
LOG_LINES = []


def log(*args):
    msg = " ".join(str(a) for a in args)
    print(f"[{(time.time() - T_START) / 60:6.1f} min] {msg}", flush=True)
    LOG_LINES.append(msg)


def selftest_formulas(device):
    g = torch.Generator().manual_seed(0)
    n = 4000

    def r(lo, hi):
        return lo + (hi - lo) * torch.rand(n, generator=g, dtype=torch.float64)

    u, v, swh, mwd, hd, vs = r(-20, 20), r(-20, 20), r(0, 8), r(0, 360), r(0, 360), r(0.2, 9)
    tws = torch.sqrt(u ** 2 + v ** 2)
    wfrom = torch.remainder(180.0 + torch.rad2deg(torch.atan2(u, v)), 360.0)
    s, c = torch.sin(torch.deg2rad(mwd)), torch.cos(torch.deg2rad(mwd))
    sh, ch = torch.sin(torch.deg2rad(hd)), torch.cos(torch.deg2rad(hd))
    w = torch.stack([u, v, swh, s, c], 1).float().to(device)
    out_w = torch.empty(n, dtype=torch.float32, device=device)
    out_n = torch.empty_like(out_w)
    _power_kernel[(n + 255) // 256, 256](_dev(w), _dev(sh.float().to(device)), _dev(ch.float().to(device)),
                                         _dev(vs.float().to(device)), _dev(out_w), _dev(out_n))
    torch.cuda.synchronize()
    for wps, out in ((True, out_w), (False, out_n)):
        ref = power_angles(tws, torch.remainder(wfrom - hd, 360.0), swh, torch.remainder(mwd - hd, 360.0), vs, wps)
        err = float((ref - power_vec(u, v, swh, s, c, sh, ch, vs, wps)).abs().max())
        assert err < 1e-3, f"power_vec mismatch {err}"
        err_k = float((out.double().cpu() - ref).abs().max() / ref.abs().max().clamp_min(1))
        assert err_k < 1e-4, f"numba power mismatch {err_k}"
    lat = torch.stack([r(-60, 60), r(-60, 60)], 1)
    lon = torch.stack([r(-180, 180), r(-180, 180)], 1)
    lon[:, 1] = lon[:, 0] + (lon[:, 1] - lon[:, 0]) * 0.02
    lat[:, 1] = lat[:, 0] + (lat[:, 1] - lat[:, 0]) * 0.02
    la, lo = torch.deg2rad(lat), torch.deg2rad(lon)
    p = torch.stack([torch.cos(la) * torch.cos(lo), torch.cos(la) * torch.sin(lo), torch.sin(la)], -1)
    dist, sh2, ch2 = leg_geometry(p)
    d_ref = haversine(lat[:, 0], lon[:, 0], lat[:, 1], lon[:, 1])
    b_ref = bearing(lat[:, 0], lon[:, 0], lat[:, 1], lon[:, 1])
    b_err = torch.remainder(torch.rad2deg(torch.atan2(sh2, ch2))[:, 0] - b_ref + 180, 360) - 180
    assert float((dist[:, 0] - d_ref).abs().max()) < 1e-3 and float(b_err.abs().max()) < 1e-6, "leg geometry mismatch"
    x = torch.randn((40, 30, 35), generator=g).float()
    ours = spline_coef(x.to(device)).double().cpu().numpy()
    ref = ndi.spline_filter(np.pad(x.numpy(), PAD, mode="edge"), order=3, output=np.float64, mode="nearest")
    err_s = float(np.abs(ours - ref)[PAD:-PAD, PAD:-PAD, PAD:-PAD].max())
    assert err_s < 1e-4, f"GPU spline prefilter mismatch {err_s}"
    log(f"formula self-tests ok (power torch/numba, leg geometry, spline prefilter err {err_s:.1e})")


class V2Tracks:
    def __init__(self):
        marker = f"{TEAM}-{V2_SUB}-AO_WPS.csv"
        zips, self.root = [], None
        for root, dirs, files in os.walk("/kaggle/input"):
            if marker in files and "tracks" in dirs:
                self.root = Path(root)
                break
            zips += [Path(root) / f for f in files if f == f"{TEAM}-{V2_SUB}.zip"]
            dirs[:] = [d for d in dirs if d != "tracks" and root.count(os.sep) < 6]
        if self.root is None:
            assert zips, "dataset namhair-2 not attached (no V2 File A CSVs and no Namhair-2.zip under /kaggle/input)"
            self.root = Path("/tmp/v2")
            if not (self.root / marker).exists():
                with zipfile.ZipFile(zips[0]) as zf:
                    zf.extractall(self.root)
        self.energy = {}
        for case in CASES:
            with open(self.root / f"{TEAM}-{V2_SUB}-{case}.csv") as f:
                for row in csv.DictReader(f):
                    d = DEPARTURES.index(datetime.strptime(row["departure_time_utc"], DTFMT))
                    self.energy[(case, d)] = float(row["energy_cons_mwh"])
        assert len(self.energy) == len(CASES) * len(DEPARTURES), "V2 File A incomplete"

    def tracks(self, case, days):
        arrs = [np.loadtxt(self.root / "tracks" / track_name(V2_SUB, case, d), delimiter=",", skiprows=1,
                           usecols=(1, 2)) for d in days]
        return np.array([a[:, 0] for a in arrs]), np.array([a[:, 1] for a in arrs])


class Run:
    def __init__(self, name, root, polys, device, smoke):
        c = CORRIDORS[name]
        self.name, self.c, self.device, self.smoke = name, c, device, smoke
        self.corr = Corridor(root, name)
        self.frame = Frame(c["src"], c["dst"])
        self.land = LandRaster(polys, self.corr, device)
        self.d_nm = self.frame.theta * EARTH_R / NM
        gc_step = SMOKE["gc_step_nm"] if smoke else GC_DP["step_nm"]
        self.gc = Lattice(self.frame, self.corr, int(round(self.d_nm / gc_step)), 0.0, 0.0, GC_DP["stage_h"],
                          GC_DP["vmax_kn"], device)
        oc = dict(O_DP)
        if smoke:
            oc.update({k: SMOKE[k] for k in ("dalpha_deg", "dbeta_deg", "beta_max_deg")})
        self.o_cfg = oc
        self.o = Lattice(self.frame, self.corr, int(round(math.degrees(self.frame.theta) / oc["dalpha_deg"])),
                         oc["dbeta_deg"], oc["beta_max_deg"], oc["stage_h"], oc["vmax_kn"], device, land=self.land,
                         node_land_km=oc["node_land_km"], edge_land_km=oc["edge_land_km"], box_margin=oc["box_margin"])
        self.bands = [self.o.band_mask(lo, hi, BAND_SPAN) for lo, hi in BANDS]
        self.env = (oc["hs_lim"], oc["tws_lim"], oc["dilate"])
        self.base = {v: baseline_points(name, VARIANTS[v][0] == "trap") for v in VARIANTS}
        self.M = c["passage_h"] * WP_PER_H
        log(f"{name}: GC {self.d_nm:.1f} nm, {self.M} legs per track")

    def tube(self, alpha, beta, t):
        step = SMOKE["tube_step_nm"] if self.smoke else t["step_nm"]
        return Lattice(self.frame, self.corr, int(round(self.d_nm / step)), step / 60.0, t["half_nm"] / 60.0,
                       t["stage_h"], t["vmax_kn"], self.device, land=self.land, node_land_km=t["node_land_km"],
                       edge_land_km=t["edge_land_km"], box_margin=t["box_margin"], centre=(alpha, beta),
                       slope_max=t["slope_max"], quiet=True)

    def pack(self, deps):
        h = [dep_hour(DEPARTURES[d]) for d in deps]
        return read_pack(self.corr, min(h) - 1, max(h) + self.c["passage_h"] + 1)


def rounded(frame, alpha, beta, c):
    lat, lon = latlon(frame.xyz(alpha, beta))
    lat = np.char.mod("%.6f", lat).astype(np.float64)
    lon = np.char.mod("%.6f", lon).astype(np.float64)
    lat[:, 0], lon[:, 0] = c["src"]
    lat[:, -1], lon[:, -1] = c["dst"]
    return lat, lon


def assess(run, win, lat, lon, t_h, variant, union=None):
    tl = lambda a: torch.tensor(a, dtype=torch.float64, device=run.device)
    m = evaluate(win, tl(lat), tl(lon), tl(t_h), variant)
    out = {k: m[k].cpu().numpy() for k in m}
    if union is not None:
        out["land"] = np.array([int(on_land(union, la, lo).sum()) for la, lo in zip(lat, lon)])
        _, fy, fx = run.corr.frac(lat, run.corr.norm_lon(lon), t_h, clip=False)
        out["box"] = ((fy < 0) | (fy > run.corr.nlat - 1) | (fx < 0) | (fx > run.corr.nlon - 1)).sum(1)
    return out


V2_KEYS = ["gc_v2w", "gc_v2n", "o_v2w", "o_v2n"]
NT_KEYS = ["gc_v2w_nt", "gc_v2n_nt", "o_v2w_nt", "o_v2n_nt", "o_v2w_ntn", "o_v2n_ntw"]
GC_KEYS = ["gc_dpw_nt", "gc_dpn_nt"]
SEED_KEYS = ([f"o_{t}{m}_nt" for t in ("prev", "next") for m in "wn"]
             + [f"o_band{k}{m}_nt" for k in range(len(BANDS)) for m in "wn"])
TUBE_KEYS = ["o_tubew_nt", "o_tuben_nt"]
SETS = [("v2", V2_KEYS), ("newton", V2_KEYS + NT_KEYS), ("gc_dp", V2_KEYS + NT_KEYS + GC_KEYS),
        ("seeds", V2_KEYS + NT_KEYS + GC_KEYS + SEED_KEYS), ("tube", V2_KEYS + NT_KEYS + GC_KEYS + SEED_KEYS + TUBE_KEYS),
        ("deep", None)]


def dp_tracks(run, lat, win, dep_h, env, ok=None):
    taus = lat.stages(dep_h, run.c["passage_h"])
    wthr = lat.node_weather(win, taus)
    if env is None:
        flags = torch.zeros((len(taus) - 1, lat.ni * lat.nj), dtype=torch.uint8, device=run.device)
    else:
        flags = lat.interval_flags(win, taus, *env)
    paths = lat.solve(wthr, flags, taus, 0.0 if env is None else run.o_cfg["big"], ok)
    del wthr, flags
    return {m: None if paths[m] is None else lat.waypoints(paths[m], taus, WP_PER_H) for m in (1, 0)}


def refine(run, win, items, t_h, free_beta, constrained, stages, trace):
    nb = len(t_h)
    A = np.concatenate([it[1] for it in items])
    B = np.concatenate([it[2] for it in items])
    W = np.concatenate([np.full(nb, float(it[3])) for it in items])
    T = np.tile(t_h, (len(items), 1))
    outs = []
    for s in range(0, len(A), NEWTON_CHUNK):
        sl = slice(s, s + NEWTON_CHUNK)
        outs.append(newton(win, run.frame, run.land, A[sl], B[sl], T[sl], W[sl], free_beta, constrained, NEWTON,
                           stages, trace))
    A, B = np.concatenate([o[0] for o in outs]), np.concatenate([o[1] for o in outs])
    return {it[0]: (A[k * nb:(k + 1) * nb], B[k * nb:(k + 1) * nb]) for k, it in enumerate(items)}


def pick(scored, keys, i, ek, opt):
    best = None
    for k in keys:
        s = scored[k][2]
        viol = 0.0
        if opt:
            viol = (max(0.0, s["max_hs"][i] - SELECT["hs_max"]) + max(0.0, s["max_tws"][i] - SELECT["tws_max"])
                    + s["land"][i] + s["box"][i])
        key = (viol > 0, viol, s[ek][i])
        if best is None or key < best[0]:
            best = (key, k)
    return best[1]


def allowed(keys, opt):
    return [k for k in keys if k.startswith("gc") or (opt and k.startswith("o"))]


def process_batch(run, win, deps, level, smoke, union, v2):
    t0 = time.time()
    frag = {"base": {}, "tracks": {}, "pick": {}, "final": {}, "sets": {},
            "trace": {"gc": {}, "o": {}, "tube": {}, "tube2": {}, "fine": {}}}
    c, M, dev, frame = run.c, run.M, run.device, run.frame
    pre = c["prefix"]
    stages = SMOKE["stages"] if smoke else NEWTON["stages"]
    dep_h = np.array([dep_hour(DEPARTURES[d]) for d in deps])
    t_h = dep_h[:, None] + np.arange(M + 1)[None] / WP_PER_H
    nb = len(deps)
    tl = lambda a: torch.tensor(a, dtype=torch.float64, device=dev)
    for v in VARIANTS:
        la, lo, offs = run.base[v]
        m = evaluate(win, tl(np.tile(la, (nb, 1))), tl(np.tile(lo, (nb, 1))), tl(dep_h[:, None] + offs[None]), v)
        for i, d in enumerate(deps):
            for cs in ("O", "GC"):
                frag["base"][(f"{pre}{cs}_WPS", d, v)] = float(m["E_wps"][i])
                frag["base"][(f"{pre}{cs}_noWPS", d, v)] = float(m["E_nowps"][i])
    modes = (("WPS", "w", 1), ("noWPS", "n", 0))
    zeros = np.zeros((nb, M + 1))
    direct, geo = {}, {}
    for cs, tag in (("GC", "gc"), ("O", "o")):
        for s, m, _ in modes:
            la, lo = v2.tracks(f"{pre}{cs}_{s}", deps)
            direct[f"{tag}_v2{m}"] = (la, lo)
            geo[f"{tag}_v2{m}"] = frame.alpha_beta(unit_vec(la, lo))
    for m in "wn":
        geo[f"gc_v2{m}"] = (geo[f"gc_v2{m}"][0], zeros)
    scored = {}

    def score(names):
        for k in names:
            if k in direct:
                lat, lon = direct[k]
            else:
                lat, lon = rounded(frame, *geo[k], c)
            scored[k] = (lat, lon, assess(run, win, lat, lon, t_h, OPT_VARIANT, union))

    score(V2_KEYS)
    frag["v2_diff"] = max(abs(scored[f"{tag}_v2{m}"][2][f"E_{'wps' if w else 'nowps'}"][i]
                              - v2.energy[(f"{pre}{cs}_{s}", d)])
                          for cs, tag in (("GC", "gc"), ("O", "o")) for s, m, w in modes for i, d in enumerate(deps))
    if level != "min":
        alpha_lin = np.linspace(0.0, frame.theta, M + 1)
        gca = {1: [], 0: []}
        for i in range(nb):
            tr = dp_tracks(run, run.gc, win, dep_h[i], None)
            for w in (1, 0):
                gca[w].append(alpha_lin if tr[w] is None else tr[w][:, 0])
        items = [("gc_v2w_nt", geo["gc_v2w"][0], zeros, 1), ("gc_v2n_nt", geo["gc_v2n"][0], zeros, 0),
                 ("gc_dpw_nt", np.array(gca[1]), zeros, 1), ("gc_dpn_nt", np.array(gca[0]), zeros, 0)]
        geo.update(refine(run, win, items, t_h, False, False, stages, frag["trace"]["gc"]))
        items = [("o_v2w_nt", *geo["o_v2w"], 1), ("o_v2n_nt", *geo["o_v2n"], 0),
                 ("o_v2w_ntn", *geo["o_v2w"], 0), ("o_v2n_ntw", *geo["o_v2n"], 1)]
        geo.update(refine(run, win, items, t_h, True, True, stages, frag["trace"]["o"]))
        score(NT_KEYS + GC_KEYS)
    frag["t_lite"] = time.time() - t0

    def incumbent(i, w):
        ek = f"E_{'wps' if w else 'nowps'}"
        return pick(scored, allowed(list(scored), True), i, ek, True)

    def tube_round(t, tag, trace):
        out = {1: [], 0: []}
        for _, m, w in modes:
            for i in range(nb):
                inc = incumbent(i, w)
                a, b = geo[inc][0][i], geo[inc][1][i]
                tr = dp_tracks(run, run.tube(a, b, t), win, dep_h[i], run.env)
                out[w].append(np.stack([a, b], 1) if tr[w] is None else tr[w])
                torch.cuda.empty_cache()
        items = [(f"o_{tag}{m}_nt", np.array(out[w])[..., 0], np.array(out[w])[..., 1], w) for _, m, w in modes]
        geo.update(refine(run, win, items, t_h, True, True, stages, trace))
        score([it[0] for it in items])

    if level in ("full", "deep"):
        items = []
        for s, m, w in modes:
            for tag, shift in (("prev", -1), ("next", 1)):
                la, lo = v2.tracks(f"{pre}O_{s}", [min(max(d + shift, 0), len(DEPARTURES) - 1) for d in deps])
                items.append((f"o_{tag}{m}_nt", *frame.alpha_beta(unit_vec(la, lo)), w))
        band = {(k, w): [] for k in range(len(BANDS)) for w in (1, 0)}
        for i in range(nb):
            taus = run.o.stages(dep_h[i], c["passage_h"])
            wthr = run.o.node_weather(win, taus)
            flags = run.o.interval_flags(win, taus, *run.env)
            for k, ok in enumerate(run.bands):
                paths = run.o.solve(wthr, flags, taus, run.o_cfg["big"], ok)
                for w, m in ((1, "w"), (0, "n")):
                    fallback = np.stack([geo[f"o_v2{m}"][0][i], geo[f"o_v2{m}"][1][i]], 1)
                    band[(k, w)].append(fallback if paths[w] is None else run.o.waypoints(paths[w], taus, WP_PER_H))
            del wthr, flags
            torch.cuda.empty_cache()
        for k in range(len(BANDS)):
            for w, m in ((1, "w"), (0, "n")):
                arr = np.array(band[(k, w)])
                items.append((f"o_band{k}{m}_nt", arr[..., 0], arr[..., 1], w))
        geo.update(refine(run, win, items, t_h, True, True, stages, frag["trace"]["o"]))
        score(SEED_KEYS)
        tube_round(TUBE, "tube", frag["trace"]["tube"])
    frag["t_full"] = time.time() - t0
    if level == "deep":
        tube_round(TUBE2, "tube2", frag["trace"]["tube2"])
        fine = SMOKE["stages"] if smoke else NEWTON_FINE
        gc_inc = {w: [pick(scored, allowed(list(scored), False), i, f"E_{'wps' if w else 'nowps'}", False)
                      for i in range(nb)] for w in (1, 0)}
        for tag, free, chooser in (("gc", False, lambda i, w: gc_inc[w][i]), ("o", True, incumbent)):
            items = []
            for _, m, w in modes:
                names = [chooser(i, w) for i in range(nb)]
                items.append((f"{tag}_fin{m}", np.array([geo[k][0][i] for i, k in enumerate(names)]),
                              np.array([geo[k][1][i] for i, k in enumerate(names)]), w))
            geo.update(refine(run, win, items, t_h, free, free, fine, frag["trace"]["fine"]))
            score([it[0] for it in items])
    cases = [f"{pre}{cs}_{s}" for cs in ("O", "GC") for s in ("WPS", "noWPS")]
    sel = {}
    for i, d in enumerate(deps):
        for case in cases:
            ek = "E_wps" if case.endswith("_WPS") else "E_nowps"
            opt = case[1] == "O"
            for name, keys in SETS:
                k = pick(scored, allowed([k for k in (keys or scored) if k in scored], opt), i, ek, opt)
                frag["sets"][(name, case, d)] = float(scored[k][2][ek][i])
            frag["tracks"][(case, d)] = (scored[k][0][i], scored[k][1][i])
            frag["pick"][(case, d)] = k
            sel[(case, d)] = (int(scored[k][2]["land"][i]), int(scored[k][2]["box"][i]))
    for case in cases:
        lat = np.array([frag["tracks"][(case, d)][0] for d in deps])
        lon = np.array([frag["tracks"][(case, d)][1] for d in deps])
        ek = "E_wps" if case.endswith("_WPS") else "E_nowps"
        for v in VARIANTS:
            s = assess(run, win, lat, lon, t_h, v)
            for i, d in enumerate(deps):
                frag["final"][(case, d, v)] = (float(s[ek][i]), float(s["max_tws"][i]), float(s["max_hs"][i]))
                frag["final"][(case, d, "validity")] = sel[(case, d)]
    return frag


def choose_level(name, jobs, jobdir, times, deadline, n_workers):
    default = {"deep": {"atlantic": 900.0, "pacific": 1600.0}, "full": {"atlantic": 650.0, "pacific": 1200.0},
               "lite": {"atlantic": 180.0, "pacific": 300.0}}
    mean = {lv: {n: (float(np.mean(times[lv][n])) if times[lv][n] else default[lv][n]) for n in CORRIDORS}
            for lv in default}
    others = [jobs[k][0] for k in range(len(jobs)) if not (jobdir / f"job{k}.lock").exists()]
    left = deadline - time.time()
    for level, base in (("deep", "full"), ("full", "full"), ("lite", "lite")):
        if mean[level][name] + sum(mean[base][n] for n in others) / n_workers < left:
            return level
    return "min"


def worker(tag, gpu):
    dev = torch.device("cuda")
    torch.set_num_threads(2)
    root = os.environ["ROUTING_DATA"]
    jobdir = WORK / tag
    jobs = json.loads((jobdir / "jobs.json").read_text())
    deadline, n_workers = float(os.environ["ROUTING_DEADLINE"]), int(os.environ["ROUTING_NW"])
    smoke = tag == "smoke"
    log(f"worker gpu{gpu} ({torch.cuda.get_device_name(0)}) tag {tag}")
    polys = read_polygons(Path(root) / "ne_10m_land.shp")
    union = land_union(polys)
    v2 = V2Tracks()
    if smoke:
        reference_check(dev)
        selftest_formulas(dev)
    runs = {}

    def claim():
        for k in range(len(jobs)):
            try:
                os.close(os.open(jobdir / f"job{k}.lock", os.O_CREAT | os.O_EXCL))
                return k
            except FileExistsError:
                continue
        return None

    def get_run(name):
        if name not in runs:
            runs[name] = Run(name, Path(root), polys, dev, smoke)
        return runs[name]

    pool = ThreadPoolExecutor(1)
    times = {lv: {n: [] for n in CORRIDORS} for lv in ("deep", "full", "lite")}
    j = claim()
    fut = pool.submit(get_run(jobs[j][0]).pack, jobs[j][1]) if j is not None else None
    while j is not None:
        name, deps = jobs[j]
        run = get_run(name)
        pack = fut.result()
        level = "deep" if smoke else choose_level(name, jobs, jobdir, times, deadline, n_workers)
        nxt = claim()
        fut = pool.submit(get_run(jobs[nxt][0]).pack, jobs[nxt][1]) if nxt is not None else None
        t0 = time.time()
        win = Window(run.corr, pack, dev)
        del pack
        t_win = time.time() - t0
        if smoke:
            errs = win.check_against_scipy()
            log(f"{name}: B-spline taps vs scipy map_coordinates max abs err {[f'{e:.1e}' for e in errs]}")
            assert max(errs[:5]) < 2e-3, "interpolation does not reproduce map_coordinates"
            err_h, err_s = newton_selftest(win, run.frame, run.land, NEWTON)
            log(f"{name}: Newton self-test ok (Hessian blocks rel err {err_h:.1e}, block solver rel err {err_s:.1e})")
        frag = process_batch(run, win, deps, level, smoke, union, v2)
        if smoke:
            lat, lon = frag["tracks"][(f"{run.c['prefix']}O_WPS", deps[0])]
            wps = [(DEPARTURES[deps[0]] + np.timedelta64(15 * k, "m").item(), la, lo)
                   for k, (la, lo) in enumerate(zip(lat, lon))]
            assert len(scorer_resample(scorer_resample(wps, 30), 15.0)) == len(wps), "15-min track is resampled"
            assert frag["v2_diff"] < 0.01, f"V2 tracks do not reproduce V2 File A energies ({frag['v2_diff']:.4f})"
        del win
        torch.cuda.empty_cache()
        dt = time.time() - t0
        if level != "min":
            times[level][name].append(dt)
        if level in ("full", "deep"):
            times["lite"][name].append(frag["t_lite"])
        if level == "deep":
            times["full"][name].append(frag["t_full"])
        frag["timing"] = (name, deps[0], level, dt, frag.get("t_lite", 0.0))
        tmp = jobdir / f"res{j}.tmp"
        with open(tmp, "wb") as f:
            pickle.dump(frag, f)
        os.replace(tmp, jobdir / f"res{j}.pkl")
        log(f"job {j + 1}/{len(jobs)} {name} days {deps[0]}-{deps[-1]} level {level}: {dt:.0f} s (lite part "
            f"{frag['t_lite']:.0f} s) (window {t_win:.0f} s, peak GPU {torch.cuda.max_memory_allocated() / 1e9:.1f} GB, "
            f"V2 check {frag['v2_diff']:.4f} MWh)")
        j = nxt
    log(f"worker gpu{gpu} done")


def run_workers(tag, jobs, root, n_gpu, deadline):
    d = WORK / tag
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    (d / "jobs.json").write_text(json.dumps(jobs))
    procs, offsets = [], []
    for g in range(n_gpu):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), ROUTING_DATA=str(root), ROUTING_DEADLINE=str(deadline),
                   ROUTING_NW=str(n_gpu), ROUTING_T0=str(T_START), PYTHONUNBUFFERED="1")
        f = open(d / f"worker{g}.log", "w")
        procs.append(subprocess.Popen([sys.executable, os.path.abspath(__file__), "worker", tag, str(g)], env=env,
                                      stdout=f, stderr=subprocess.STDOUT))
        offsets.append(0)

    def stream():
        for g in range(n_gpu):
            with open(d / f"worker{g}.log") as f:
                f.seek(offsets[g])
                text = f.read()
                offsets[g] = f.tell()
            for line in text.splitlines():
                print(f"  gpu{g} | {line}", flush=True)
                LOG_LINES.append(f"gpu{g} | {line}")

    hard = deadline + (0 if tag == "smoke" else 600)
    while any(p.poll() is None for p in procs):
        stream()
        if time.time() > hard:
            log(f"{tag}: hard deadline reached, killing workers")
            for p in procs:
                p.kill()
            break
        time.sleep(15)
    stream()
    log(f"{tag} workers exit codes {[p.wait() for p in procs]}")


def absorb(store, frag):
    for k in ("base", "tracks", "pick", "final", "sets"):
        store[k].update(frag[k])
    store["v2_diff"] = max(store["v2_diff"], frag["v2_diff"])
    for g, tr in frag["trace"].items():
        acc = store["trace"].setdefault(g, {})
        for step, (n, e) in tr.items():
            n0, e0 = acc.get(step, (0, 0.0))
            acc[step] = (n0 + n, e0 + e)


def merge(tag, jobs):
    store = {"base": {}, "tracks": {}, "pick": {}, "final": {}, "sets": {}, "trace": {}, "timing": [], "v2_diff": 0.0}
    missing = []
    for j in range(len(jobs)):
        p = WORK / tag / f"res{j}.pkl"
        if not p.exists():
            missing.append(j)
            continue
        with open(p, "rb") as f:
            frag = pickle.load(f)
        absorb(store, frag)
        store["timing"].append(frag["timing"])
    return store, missing


def summarize(store, deps_by_corr, tag):
    rows, sets = [], {name: {} for name, _ in SETS}
    for name, deps in deps_by_corr.items():
        pre = CORRIDORS[name]["prefix"]
        for case in [f"{pre}{cs}_{s}" for cs in ("O", "GC") for s in ("WPS", "noWPS")]:
            e = sum(store["final"][(case, d, OPT_VARIANT)][0] for d in deps)
            b = sum(store["base"][(case, d, OPT_VARIANT)] for d in deps)
            picks = {}
            for d in deps:
                picks[store["pick"][(case, d)]] = picks.get(store["pick"][(case, d)], 0) + 1
            rows.append(f"  {case:10s} {e:12.1f} vs baseline {b:12.1f} ({100 * (e / b - 1):+.1f}%) picks {picks}")
            for sname in sets:
                sets[sname][case] = sum(store["sets"][(sname, case, d)] for d in deps)
    log(f"{tag}:\n" + "\n".join(rows))
    log(f"V2 tracks vs V2 File A energies: max abs diff {store['v2_diff']:.4f} MWh")
    log("candidate sets (each adds one step; v2 = the V2 submission):\n" + "\n".join(
        f"  {sname:8s} total {sum(per.values()):11.1f}  " + " ".join(f"{k}:{x:.1f}" for k, x in per.items())
        for sname, per in sets.items()))
    for g, tr in store["trace"].items():
        log(f"Newton trace {g} (iteration: feasible tracks, sum of their best exact energies): "
            + ", ".join(f"{k}: {n} {e:.1f}" for k, (n, e) in sorted(tr.items())))


def calibrate(store):
    table, best = [], None
    for v in VARIANTS:
        diffs = {case: sum(store["base"][(case, d, v)] for d in range(len(DEPARTURES))) - BOARD_BASELINE[case]
                 for case in BOARD_BASELINE}
        err = max(abs(x) for x in diffs.values())
        table.append((v, err, diffs))
        if best is None or err < best[1]:
            best = (v, err)
    lines = [f"  {v:22s} max|diff| {err:10.3f}  " + " ".join(f"{k}:{x:+.2f}" for k, x in diffs.items())
             for v, err, diffs in table]
    log("calibration vs board baseline row:\n" + "\n".join(lines))
    log(f"chosen scorer variant: {best[0]} (max abs diff {best[1]:.3f} MWh per case)"
        + ("" if best[1] < 1.0 else "  WARNING: no variant reproduces the board within 1 MWh"))
    return best[0], table


def package(store, variant, deps, union, root, zip_path):
    report, n_lim = {}, 0
    for (case, d) in store["tracks"]:
        e, mw, mh = store["final"][(case, d, variant)]
        report[(case, d)] = (e, mw, mh)
        n_lim += ("GC" not in case) and (mw > MAX_WIND or mh > MAX_HS)
    if root.exists():
        shutil.rmtree(root)
    write_submission(root, store["tracks"], report, SUBMISSION_NO, deps)
    errors = validate(root, SUBMISSION_NO, deps, union)
    log(f"validation of {root}: {len(errors)} errors ({n_lim} self-reported limit breaches)")
    for err in errors[:30]:
        log("  " + err)
    assert make_zip(root, zip_path) == 8, "File A CSVs must sit at the zip root"
    return errors


def phase(tag, jobs, root, polys, n_gpu, deadline, v2):
    run_workers(tag, jobs, root, n_gpu, deadline)
    store, missing = merge(tag, jobs)
    if missing:
        log(f"{tag}: {len(missing)} jobs unfinished by workers, completing them at level 'min'")
        dev = torch.device("cuda")
        union = land_union(polys)
        runs = {}
        for j in missing:
            name, deps = jobs[j]
            if name not in runs:
                runs[name] = Run(name, root, polys, dev, tag == "smoke")
            win = Window(runs[name].corr, runs[name].pack(deps), dev)
            frag = process_batch(runs[name], win, deps, "min", tag == "smoke", union, v2)
            del win
            absorb(store, frag)
        torch.cuda.empty_cache()
    return store


def orchestrate():
    WORK.mkdir(parents=True, exist_ok=True)
    n_gpu = torch.cuda.device_count()
    assert n_gpu >= 1, "GPU notebook required"
    root = find_era5_dir()
    polys = read_polygons(root / "ne_10m_land.shp")
    union = land_union(polys)
    v2 = V2Tracks()
    log(f"data {root}; V2 tracks {v2.root}; {len(polys)} land polygons; {n_gpu} GPU(s)")

    smoke_jobs = [(name, SMOKE["days"]) for name in ("pacific", "atlantic")]
    store = phase("smoke", smoke_jobs, root, polys, n_gpu, time.time() + 2400, v2)
    summarize(store, {n: SMOKE["days"] for n in CORRIDORS}, "smoke totals (2 departures)")
    package(store, OPT_VARIANT, SMOKE["days"], union, WORK / "smoke_sub", WORK / "smoke.zip")
    log("smoke ok")

    jobs = [(name, list(range(s, min(s + BATCH_DAYS, len(DEPARTURES)))))
            for name in ("pacific", "atlantic") for s in range(0, len(DEPARTURES), BATCH_DAYS)]
    store = phase("full", jobs, root, polys, n_gpu, T_START + (BUDGET_H - RESERVE_H) * 3600, v2)
    all_deps = list(range(len(DEPARTURES)))
    summarize(store, {n: all_deps for n in CORRIDORS}, "final totals (optimiser variant)")
    variant, table = calibrate(store)
    total = sum(store["final"][(case, d, variant)][0] for (case, d) in store["tracks"])
    log(f"projected leaderboard total ({variant}): {total:.1f} MWh (V2 LB 489,616.60; baseline 712,247)")
    zip_name = f"{TEAM}-{SUBMISSION_NO}.zip"
    errors = package(store, variant, all_deps, union, WORK / "submission", OUT / zip_name)

    with open(WORK / "per_departure.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case", "departure", "pick", "file"] + [f"E_{v}" for v in VARIANTS]
                   + [f"base_{v}" for v in VARIANTS] + ["max_tws", "max_hs", "land", "box"]
                   + [f"set_{name}" for name, _ in SETS])
        for (case, d) in sorted(store["tracks"]):
            fin = store["final"][(case, d, variant)]
            w.writerow([case, DEPARTURES[d].date(), store["pick"][(case, d)], track_name(SUBMISSION_NO, case, d)]
                       + [f"{store['final'][(case, d, v)][0]:.4f}" for v in VARIANTS]
                       + [f"{store['base'][(case, d, v)]:.4f}" for v in VARIANTS] + list(fin[1:])
                       + list(store["final"][(case, d, "validity")])
                       + [f"{store['sets'][(name, case, d)]:.4f}" for name, _ in SETS])
    (WORK / "calibration.json").write_text(json.dumps({"chosen": variant, "table": table}, indent=1))
    (WORK / "timing.json").write_text(json.dumps({"jobs": store["timing"], "errors": errors[:200]}, indent=1))
    (WORK / "log.txt").write_text("\n".join(LOG_LINES))
    with zipfile.ZipFile(OUT / "results.zip", "w", zipfile.ZIP_DEFLATED) as zf:
        for fn in ("log.txt", "per_departure.csv", "calibration.json", "timing.json"):
            zf.write(WORK / fn, fn)
    log(f"SUBMIT: {zip_name}  (projected {total:.1f} MWh, {len(errors)} validation errors)")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        worker(sys.argv[2], sys.argv[3])
    else:
        orchestrate()
