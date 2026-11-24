import math

import torch

K_H = 969.0 / 226.0
K_A = 49.0 / 320.0
A_W = 11.1395
K_W = 125.0 / 432.0
K_S = 27489.0 / 32000.0
DEAD_DEG = 10.0


def power_angles(tws, twa_deg, swh, mwa_deg, v, wps):
    twa = torch.deg2rad(twa_deg)
    ux = tws * torch.cos(twa) + v
    uy = tws * torch.sin(twa)
    vr = torch.sqrt(ux ** 2 + uy ** 2)
    power = K_H * v ** 3 + K_A * v * (vr * ux - v ** 2)
    mwa = torch.deg2rad(torch.remainder(mwa_deg + 180.0, 360.0) - 180.0)
    power = power + A_W * swh ** 2 * v ** 1.5 * torch.exp(-K_W * torch.abs(mwa) ** 3)
    if wps:
        awa = torch.rad2deg(torch.atan2(torch.abs(uy), ux))
        active = awa >= DEAD_DEG
        sa = torch.sin(torch.where(active, torch.deg2rad(awa - DEAD_DEG), torch.zeros_like(awa)))
        p_sail = K_S * sa * (1.0 + 0.15 * sa ** 2) * vr ** 2 * v
        power = power - torch.where(active, p_sail, torch.zeros_like(p_sail))
    return torch.clamp_min(power, 0.0)


def power_vec(u, v10, swh, s, c, sh, ch, vs, wps, smooth=None):
    ux = vs - (u * sh + v10 * ch)
    uy = v10 * sh - u * ch
    vr2 = ux * ux + uy * uy
    vr = torch.sqrt(vr2 + 1e-12)
    power = K_H * vs ** 3 + K_A * vs * (vr * ux - vs * vs)
    th = torch.atan2(s * ch - c * sh, c * ch + s * sh)
    power = power + A_W * swh * swh * vs * torch.sqrt(vs + 1e-12) * torch.exp(-K_W * torch.abs(th) ** 3)
    awa = torch.atan2(torch.abs(uy), ux)
    alpha = awa - math.radians(DEAD_DEG)
    sa = torch.sin(torch.clamp_min(alpha, 0.0))
    p_sail = torch.where(alpha >= 0, K_S * sa * (1.0 + 0.15 * sa * sa) * vr2 * vs, torch.zeros_like(vr2))
    if not torch.is_tensor(wps):
        wps = float(wps)
    exact = torch.clamp_min(power - wps * p_sail, 0.0)
    if smooth is None:
        return exact
    tau, dz = smooth
    awa_s = torch.atan2(torch.sqrt(uy * uy + 1e-4), ux)
    sa_s = torch.sin(torch.nn.functional.softplus(awa_s - math.radians(DEAD_DEG), beta=1.0 / dz, threshold=20.0))
    p_sail_s = K_S * sa_s * (1.0 + 0.15 * sa_s * sa_s) * vr2 * vs
    surrogate = torch.nn.functional.softplus(power - wps * p_sail_s, beta=1.0 / tau, threshold=20.0)
    return surrogate, exact
