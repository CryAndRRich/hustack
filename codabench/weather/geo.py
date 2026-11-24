import math

import numpy as np
import torch

from config import EARTH_R


def unit_vec(lat_deg, lon_deg):
    la, lo = np.radians(lat_deg), np.radians(lon_deg)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)


class Frame:
    def __init__(self, src, dst):
        p1, p2 = unit_vec(*src), unit_vec(*dst)
        n = np.cross(p1, p2)
        self.n = n / np.linalg.norm(n)
        self.e1 = p1
        self.e2 = np.cross(self.n, p1)
        self.theta = math.acos(float(np.clip(p1 @ p2, -1.0, 1.0)))

    def xyz(self, alpha, beta):
        mod = torch if torch.is_tensor(alpha) else np
        ca, sa, cb, sb = mod.cos(alpha), mod.sin(alpha), mod.cos(beta), mod.sin(beta)
        basis = [self.e1, self.e2, self.n]
        if torch.is_tensor(alpha):
            basis = [torch.as_tensor(b, dtype=alpha.dtype, device=alpha.device) for b in basis]
        return (cb * ca)[..., None] * basis[0] + (cb * sa)[..., None] * basis[1] + sb[..., None] * basis[2]

    def alpha_beta(self, p):
        return np.arctan2(p @ self.e2, p @ self.e1), np.arcsin(np.clip(p @ self.n, -1.0, 1.0))


def latlon(p):
    mod = torch if torch.is_tensor(p) else np
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    return mod.rad2deg(mod.arctan2(z, mod.sqrt(x * x + y * y))), mod.rad2deg(mod.arctan2(y, x))


def leg_geometry(p, eps=1e-16):
    a, b = p[..., :-1, :], p[..., 1:, :]
    chord = torch.sqrt(((b - a) ** 2).sum(-1) + eps)
    dist = 2.0 * EARTH_R * torch.asin(torch.clamp(chord * 0.5, max=1.0))
    d = b - (a * b).sum(-1, keepdim=True) * a
    x, y, z = a[..., 0], a[..., 1], a[..., 2]
    rho = torch.sqrt(x * x + y * y).clamp_min(1e-12)
    de = (-y * d[..., 0] + x * d[..., 1]) / rho
    dn = (-z * x * d[..., 0] - z * y * d[..., 1]) / rho + rho * d[..., 2]
    dl = torch.sqrt(de * de + dn * dn + eps)
    return dist, de / dl, dn / dl
