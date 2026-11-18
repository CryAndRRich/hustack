from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import scipy.sparse
import scipy.sparse.linalg

PERMC = "MMD_AT_PLUS_A"
BLOCK = 64

DEFAULT_SHIFTS = ((1.0, 2), (0.0, 1), (1.0 / 12.0, 1), (1.0 / 288.0, 1))


@dataclass
class Core:
    a: scipy.sparse.csc_matrix
    cap: np.ndarray
    amb: np.ndarray
    mass_coupling: np.ndarray
    mass_retained: np.ndarray
    mass_gained: np.ndarray
    applied: scipy.sparse.csc_matrix
    sensed: scipy.sparse.csr_matrix
    zones: scipy.sparse.csc_matrix
    temp0: np.ndarray
    mass0: np.ndarray
    zone_names: list[str]
    h: float

    @property
    def n(self) -> int:
        return self.a.shape[0]


def solve_block(lu, rhs) -> np.ndarray:
    n_cols = rhs.shape[1]
    out = np.empty((rhs.shape[0], n_cols))
    for start in range(0, n_cols, BLOCK):
        chunk = rhs[:, start:start + BLOCK]
        chunk = chunk.toarray() if scipy.sparse.issparse(chunk) else np.asarray(chunk)
        out[:, start:start + BLOCK] = lu.solve(np.asfortranarray(chunk))
    return out


class Augmented:
    def __init__(self, core: Core):
        self.core = core
        self.n = core.n
        self.mi = np.flatnonzero(core.mass_coupling > 0.0)
        mc = core.mass_coupling[self.mi]
        self.g_m = mc / core.mass_retained[self.mi]
        self.c_m = mc / core.mass_gained[self.mi]
        self.N = self.n + len(self.mi)
        self.e = np.concatenate([core.cap, self.c_m])
        self.sqrt_e = np.sqrt(self.e)
        self.k_air = (core.a - scipy.sparse.diags(core.cap + core.mass_coupling)).tocsr()

    def k_mul(self, x: np.ndarray) -> np.ndarray:
        n, mi, g = self.n, self.mi, self.g_m[:, None]
        xt, xm = x[:n], x[n:]
        out_t = np.asarray(self.k_air @ xt)
        diff = xt[mi] - xm
        out_t[mi] += g * diff
        return np.vstack([out_t, -g * diff])

    def shifted_solver(self, sigma: float):
        n, mi = self.n, self.mi
        dm = sigma * self.c_m + self.g_m
        if sigma == 1.0:
            a_s = self.core.a
        else:
            extra = np.zeros(n)
            extra[mi] = sigma * self.c_m * self.g_m / dm
            a_s = (self.k_air + scipy.sparse.diags(sigma * self.core.cap + extra)).tocsc()
        lu = scipy.sparse.linalg.splu(a_s, permc_spec=PERMC)

        def solve(b: np.ndarray) -> np.ndarray:
            bt = np.array(b[:n], dtype=np.float64)
            bm = b[n:]
            bt[mi] += (self.g_m / dm)[:, None] * bm
            zt = solve_block(lu, bt)
            zm = (bm + self.g_m[:, None] * zt[mi]) / dm[:, None]
            return np.vstack([zt, zm])

        return solve, lu

    def inputs(self) -> scipy.sparse.csc_matrix:
        c = self.core
        top = scipy.sparse.hstack([c.zones, c.applied, scipy.sparse.csc_matrix(c.amb[:, None])])
        return scipy.sparse.vstack([top, scipy.sparse.csc_matrix((len(self.mi), top.shape[1]))]).tocsc()

    def state0(self, t_ref: float) -> np.ndarray:
        return np.concatenate([self.core.temp0, self.core.mass0[self.mi]]) - t_ref


class Basis:
    def __init__(self, sqrt_e: np.ndarray, capacity: int):
        self.sqrt_e = sqrt_e
        self.b = np.empty((len(sqrt_e), capacity))
        self.r = 0
        self.dropped = 0

    def _project_out(self, w):
        if self.r:
            cur = self.b[:, :self.r]
            w -= cur @ (cur.T @ w)
        return w

    def extend(self, z: np.ndarray, tol: float) -> int:
        w = z * self.sqrt_e[:, None]
        norms = np.linalg.norm(w, axis=0)
        w = w[:, norms > 0.0] / norms[norms > 0.0]
        for _ in range(2):
            w = self._project_out(w)
        mu, vec = np.linalg.eigh(w.T @ w)
        keep = mu > tol ** 2
        self.dropped += int(np.sum(~keep)) + int(np.sum(norms == 0.0))
        w = w @ (vec[:, keep] / np.sqrt(mu[keep]))
        w = self._project_out(w)
        mu, vec = np.linalg.eigh(w.T @ w)
        keep = mu > 1e-2
        self.dropped += int(np.sum(~keep))
        w = w @ (vec[:, keep] / np.sqrt(mu[keep]))
        k = w.shape[1]
        self.b[:, self.r:self.r + k] = w
        self.r += k
        return k

    @property
    def cols(self) -> np.ndarray:
        return self.b[:, :self.r]

    def v_chunks(self, size: int = 256):
        for start in range(0, self.r, size):
            sl = slice(start, min(start + size, self.r))
            yield sl, self.b[:, sl] / self.sqrt_e[:, None]


def build_basis(aug: Augmented, f: scipy.sparse.csc_matrix, x0: np.ndarray, shifts=DEFAULT_SHIFTS,
                tol: float = 1e-6, log=print) -> tuple[Basis, dict]:
    rhs = f.toarray()
    if np.linalg.norm(x0 * aug.sqrt_e) > 0.0:
        rhs = np.hstack([rhs, (aug.e * x0)[:, None]])
    basis = Basis(aug.sqrt_e, rhs.shape[1] * sum(d for _, d in shifts))
    info = {"blocks": []}
    for sigma, depth in shifts:
        t0 = time.time()
        solve, lu = aug.shifted_solver(sigma)
        t_lu = time.time() - t0
        z = rhs
        for j in range(depth):
            z = solve(z if j == 0 else aug.e[:, None] * z)
            added = basis.extend(z, tol)
            info["blocks"].append({"sigma": sigma, "moment": j + 1, "added": added, "r": basis.r,
                                   "lu_s": round(t_lu, 2), "s": round(time.time() - t0, 2)})
            log(f"  basis sigma={sigma:.5f} moment={j + 1}: +{added} -> r={basis.r} ({time.time() - t0:.1f}s)")
        del lu, solve
    info["dropped"] = basis.dropped
    return basis, info


@dataclass
class LinearRom:
    a: np.ndarray
    b: np.ndarray
    c: np.ndarray
    x0: np.ndarray
    y0: np.ndarray
    t_ref: float
    h: float
    n_zones: int
    n_vavs: int

    def __post_init__(self):
        z, v = self.n_zones, self.n_vavs
        self._np = self.c @ self.b[:, z:z + v]
        self._m = self._np[:v]

    @property
    def order(self) -> int:
        return len(self.x0)


def reduce(aug: Augmented, basis: Basis, f: scipy.sparse.csc_matrix, x0: np.ndarray) -> dict[str, np.ndarray]:
    r, n = basis.r, aug.n
    b = basis.cols
    k_r = np.empty((r, r))
    for sl, v in basis.v_chunks():
        k_r[:, sl] = b.T @ (aug.k_mul(v) / aug.sqrt_e[:, None])
    f_r = np.asarray((scipy.sparse.diags(1.0 / aug.sqrt_e) @ f).T @ b).T
    x0_r = b.T @ (aug.sqrt_e * x0)
    c = aug.core
    out_t = scipy.sparse.vstack([c.sensed, c.zones.T]).tocsr() @ scipy.sparse.diags(1.0 / aug.sqrt_e[:n])
    c_r = np.asarray(out_t @ b[:n])
    return {"k": 0.5 * (k_r + k_r.T), "k_asym": float(np.abs(k_r - k_r.T).max() / np.abs(k_r).max()),
            "e_err": float(np.abs(b.T @ b - np.eye(r)).max()), "f": f_r, "c": c_r, "x0": x0_r}


def modal_rom(red: dict, core: Core, t_ref: float) -> tuple[LinearRom, dict]:
    lam, w = np.linalg.eigh(red["k"])
    info = {"lambda_raw_min": float(lam.min())}
    lam = np.maximum(lam, 1e-12)
    info["slowest_time_constant_steps"] = float(1.0 / lam.min())
    a = 1.0 / (1.0 + lam)
    b = a[:, None] * (w.T @ red["f"])
    y0 = np.concatenate([core.sensed @ core.temp0, core.zones.T @ core.temp0])
    rom = LinearRom(a=a, b=b, c=red["c"] @ w, x0=w.T @ red["x0"], y0=y0, t_ref=t_ref, h=core.h,
                    n_zones=len(core.zone_names), n_vavs=core.applied.shape[1])
    return rom, info


def build_rom(core: Core, shifts=DEFAULT_SHIFTS, tol: float = 1e-6, log=None) -> tuple[LinearRom, dict]:
    log = log or (lambda s: None)
    t0 = time.time()
    t_ref = float(np.mean(core.temp0))
    aug = Augmented(core)
    f = aug.inputs()
    x0 = aug.state0(t_ref)
    basis, basis_info = build_basis(aug, f, x0, shifts, tol, log)
    t1 = time.time()
    red = reduce(aug, basis, f, x0)
    del basis
    rom, modal_info = modal_rom(red, core, t_ref)
    info = {"basis": basis_info, "e_r_err": red["e_err"], "k_r_asym": red["k_asym"], "n_mass": len(aug.mi),
            "n_aug": aug.N, "x0_dev_max_k": float(np.abs(x0).max()), "basis_s": t1 - t0, "reduce_s": time.time() - t1,
            **modal_info}
    return rom, info
