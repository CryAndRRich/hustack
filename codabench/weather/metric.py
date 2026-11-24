import math
from datetime import timedelta

import numpy as np
import torch

from config import CORRIDORS, DEPARTURES, T0, VARIANTS, MAX_HS, MAX_WIND
from rise import power_angles


def scorer_slerp(lat1, lon1, lat2, lon2, f):
    phi1, lam1 = math.radians(lat1), math.radians(lon1)
    phi2, lam2 = math.radians(lat2), math.radians(lon2)
    p1 = np.array([math.cos(phi1) * math.cos(lam1), math.cos(phi1) * math.sin(lam1), math.sin(phi1)])
    p2 = np.array([math.cos(phi2) * math.cos(lam2), math.cos(phi2) * math.sin(lam2), math.sin(phi2)])
    dot = float(np.clip(np.dot(p1, p2), -1.0, 1.0))
    sigma = math.acos(dot)
    if sigma < 1e-12:
        return lat1 + f * (lat2 - lat1), lon1 + f * ((lon2 - lon1 + 180) % 360 - 180)
    a = math.sin((1 - f) * sigma) / math.sin(sigma)
    b = math.sin(f * sigma) / math.sin(sigma)
    p = a * p1 + b * p2
    return math.degrees(math.atan2(p[2], math.sqrt(p[0] ** 2 + p[1] ** 2))), math.degrees(math.atan2(p[1], p[0]))


def scorer_resample(waypoints, dt_minutes):
    result = []
    for i in range(len(waypoints) - 1):
        t0, lat0, lon0 = waypoints[i]
        t1, lat1, lon1 = waypoints[i + 1]
        seg_seconds = (t1 - t0).total_seconds()
        if seg_seconds <= 0:
            result.append((t0, lat0, lon0))
            continue
        n_sub = max(1, math.ceil(seg_seconds / (dt_minutes * 60.0)))
        for j in range(n_sub):
            f = j / n_sub
            lat, lon = scorer_slerp(lat0, lon0, lat1, lon1, f)
            result.append((t0 + timedelta(seconds=f * seg_seconds), lat, lon))
    result.append(waypoints[-1])
    return result


def kit_waypoints(corridor, departure, n_points=100):
    c = CORRIDORS[corridor]
    (la1, lo1), (la2, lo2) = c["src"], c["dst"]
    lat1, lon1, lat2, lon2 = map(math.radians, (la1, lo1, la2, lo2))
    d = math.acos(math.sin(lat1) * math.sin(lat2) + math.cos(lat1) * math.cos(lat2) * math.cos(lon2 - lon1))
    total = c["passage_h"] * 3600
    wps = []
    for i in range(n_points):
        f = i / (n_points - 1)
        a, b = math.sin((1 - f) * d) / math.sin(d), math.sin(f * d) / math.sin(d)
        x = a * math.cos(lat1) * math.cos(lon1) + b * math.cos(lat2) * math.cos(lon2)
        y = a * math.cos(lat1) * math.sin(lon1) + b * math.cos(lat2) * math.sin(lon2)
        z = a * math.sin(lat1) + b * math.sin(lat2)
        lat, lon = math.degrees(math.atan2(z, math.sqrt(x ** 2 + y ** 2))), math.degrees(math.atan2(y, x))
        t = (departure + timedelta(seconds=total * i / (n_points - 1))).replace(microsecond=0)
        wps.append((t, float(f"{lat:.6f}"), float(f"{lon:.6f}")))
    return wps


def baseline_points(corridor, double):
    dep = DEPARTURES[0]
    wps = kit_waypoints(corridor, dep)
    pts = scorer_resample(scorer_resample(wps, 30) if double else wps, 15.0)
    times = np.array([np.datetime64(p[0]) for p in pts], dtype="datetime64[s]")
    seg_h = ((times[1:] - times[:-1]) / np.timedelta64(1, "h")).astype(np.float64)
    seg_h = np.maximum(seg_h, 1e-6)
    offs = np.concatenate([[0.0], np.cumsum(seg_h)])
    return np.array([p[1] for p in pts]), np.array([p[2] for p in pts]), offs


def dep_hour(departure):
    return (departure - T0).total_seconds() / 3600.0


def haversine(lat1, lon1, lat2, lon2):
    dlat, dlon = torch.deg2rad(lat2 - lat1), torch.deg2rad(lon2 - lon1)
    a = torch.sin(dlat / 2) ** 2 + torch.cos(torch.deg2rad(lat1)) * torch.cos(torch.deg2rad(lat2)) * torch.sin(dlon / 2) ** 2
    return 6371000.0 * 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))


def bearing(lat1, lon1, lat2, lon2):
    l1, l2, dlon = torch.deg2rad(lat1), torch.deg2rad(lat2), torch.deg2rad(lon2 - lon1)
    x = torch.sin(dlon) * torch.cos(l2)
    y = torch.cos(l1) * torch.sin(l2) - torch.sin(l1) * torch.cos(l2) * torch.cos(dlon)
    return torch.remainder(torch.rad2deg(torch.atan2(x, y)), 360.0)


def evaluate(win, lat, lon, t_h, variant):
    rule, order, mwd_mode = VARIANTS[variant]
    corr = win.corr
    lon = corr.norm_lon(lon)
    dt_h = torch.clamp_min(t_h[:, 1:] - t_h[:, :-1], 1e-6)
    vs = haversine(lat[:, :-1], lon[:, :-1], lat[:, 1:], lon[:, 1:]) / (dt_h * 3600.0)
    brg = bearing(lat[:, :-1], lon[:, :-1], lat[:, 1:], lon[:, 1:])
    if rule == "mid":
        lat, lon, t_h = (lat[:, :-1] + lat[:, 1:]) / 2, (lon[:, :-1] + lon[:, 1:]) / 2, t_h[:, :-1] + dt_h / 2
    w = (win.cubic if order == 3 else win.linear)(*corr.frac(lat, lon, t_h))
    u, v, swh = w[..., 0], w[..., 1], w[..., 2]
    mwd = w[..., 5] if mwd_mode == "raw" else torch.remainder(torch.rad2deg(torch.atan2(w[..., 3], w[..., 4])), 360.0)
    tws = torch.sqrt(u ** 2 + v ** 2)
    wfrom = torch.remainder(180.0 + torch.rad2deg(torch.atan2(u, v)), 360.0)
    out = {"max_tws": tws.max(1).values, "max_hs": swh.max(1).values,
           "n_tws": (tws > MAX_WIND).sum(1), "n_hs": (swh > MAX_HS).sum(1)}
    for wps in (True, False):
        def pw(sl):
            return power_angles(tws[:, sl], torch.remainder(wfrom[:, sl] - brg, 360.0), swh[:, sl],
                                torch.remainder(mwd[:, sl] - brg, 360.0), vs, wps)
        if rule == "mid":
            e = (pw(slice(None)) * dt_h).sum(1) / 1000.0
        else:
            e = ((pw(slice(None, -1)) + pw(slice(1, None))) / 2.0 * dt_h).sum(1) / 1000.0
        out["E_wps" if wps else "E_nowps"] = e
    return out
