import math
import threading
import time
import traceback

import numpy as np
import torch

from geometry import N_ELLIPSES, SHAPE_AB
from seeds import ASPECT_FAN, _point_pairs, hex67, make_population

WALL_TARGET, WALL_ACCEPT = 3e-4, 1e-4
MU_TARGET, MU_ACCEPT = 1e-4, 3e-5
PAIR_W = 4.0

GAMMA0, GAMMA_MIN, GAMMA_MAX = 2e-3, 1e-4, 3e-2
GAMMA_UP, GAMMA_DOWN = 1.25, 0.55
ASPECT_JITTER_REL = 2.0
ASPECT_JITTER_MAX = 4e-3
KICK = 1.01

CULL_K = 16
KNN_EVERY = 40

BB_LO, BB_HI = 0.05, 20.0

LBFGS_HISTORY = 4

AREA_SCALE = 1e-5

LR_CELL = 5e-5

GLS_UP, GLS_DECAY, GLS_MAX = 1.4, 0.94, 12.0

STALL_ESCAPE, STALL_RESET = 12, 40
ESCAPE_GRACE = 25
RELOCATE_CAND = 24
DEDUP_REL = 1e-6

WARM_SLACK = 1.00005

ADAPTIVE_FAN_EPSILON = 0.3
ADAPTIVE_FAN_WIDTH = 1.5

RESERVE_FRAC = 0.25

SHAPE_TUNE = {
    "2:1": {"stall_escape": 2 * STALL_ESCAPE, "stall_reset": 2 * STALL_RESET,
            "gls_max": GLS_MAX / 2.0, "cull_k": 10},
    "3:2": {"cull_k": 8},
}

DEFAULT_REPLICAS = {"2:1": 192, "3:2": 256, "3:1": 320}

CULL_GROUPS = (("2:1", "3:2"), ("3:1",))

CROSSOVER_FRAC = 0.4
CROSSOVER_ELITE_K = 8
CROSSOVER_INFLATE = 1.15

SA_EVERY_MULT = 4
SA_TARGET_FRAC = 0.12
SA_REMOVE_K_MAX = 4
SA_ROUNDS = 5
SA_T_HIGH, SA_T_LOW = 0.05, 0.001
SA_KICK_RANGE = (0.02, 0.08)

CONTACT_MU = 1.002
RATTLER_SHELL = 6
RATTLER_K = 2

FREEZE_GROUP_RANGE = (8, 14)
FREEZE_EVERY_MULT = 2
FREEZE_ROUNDS = 3
SYM_EVERY_MULT = 5

BOUNDARY_DEPTH = 2.0
BOUNDARY_FRAC = 0.2
BOUNDARY_LR_TH_MULT = 2.0
BOUNDARY_LR_POS_MULT = 0.6

WALL_TOUCH_SLACK = 1e-3
JAM_FREE_K = 4
JAM_ROUNDS = 5
JAM_FLIP_PROB = 0.5
JAM_POS_SIGMA = 0.06
JAM_EVERY_MULT = 3
JAM_SLACK = 1.004

LOOSEN_STRENGTH = 0.12


def _pw_F_t(lam, Ai, Aj, rx, ry):
    a11, a12, a22 = (t.unsqueeze(-1) for t in Ai)
    b11, b12, b22 = (t.unsqueeze(-1) for t in Aj)
    rx_, ry_ = rx.unsqueeze(-1), ry.unsqueeze(-1)
    m11 = a11 + lam * (b11 - a11)
    m12 = a12 + lam * (b12 - a12)
    m22 = a22 + lam * (b22 - a22)
    det = m11 * m22 - m12 * m12
    num = m22 * rx_ * rx_ - 2.0 * m12 * rx_ * ry_ + m11 * ry_ * ry_
    return lam * (1.0 - lam) * num / det

_LAM0_CACHE = {}
_FRAC_CACHE = {}


def _lam0(coarse, dtype, device):
    key = (coarse, dtype, device)
    t = _LAM0_CACHE.get(key)
    if t is None:
        t = torch.linspace(1e-4, 1.0 - 1e-4, coarse, device=device, dtype=dtype)
        _LAM0_CACHE[key] = t
    return t


def _frac_const(dtype, device):
    key = (dtype, device)
    t = _FRAC_CACHE.get(key)
    if t is None:
        t = torch.tensor([0.25, 0.5, 0.75], device=device, dtype=dtype)
        _FRAC_CACHE[key] = t
    return t


def _lambda_star_t(Ai, Aj, rx, ry, coarse, refine):
    lam = _lam0(coarse, rx.dtype, rx.device)
    vals = _pw_F_t(lam.view(1, 1, -1), Ai, Aj, rx, ry)
    c = lam[vals.argmax(dim=-1)]
    step = float(lam[1] - lam[0])
    lo = (c - step).clamp(1e-6, 1.0 - 1e-6)
    hi = (c + step).clamp(1e-6, 1.0 - 1e-6)
    frac = _frac_const(rx.dtype, rx.device)
    for _ in range(refine):
        pts = lo.unsqueeze(-1) + (hi - lo).unsqueeze(-1) * frac
        vals = _pw_F_t(pts, Ai, Aj, rx, ry)
        c = pts.gather(-1, vals.argmax(dim=-1, keepdim=True)).squeeze(-1)
        half = (hi - lo) * 0.25
        lo = (c - half).clamp(1e-6, 1.0 - 1e-6)
        hi = (c + half).clamp(1e-6, 1.0 - 1e-6)
    return c


def _pdist2_t(uv):
    n = uv.shape[1]
    diff = uv.unsqueeze(2) - uv.unsqueeze(1)
    d2 = (diff * diff).sum(-1)
    return d2 + torch.eye(n, device=uv.device, dtype=uv.dtype) * 1e9


def _knn_pairs_t(uv, th, a, b, k):
    B, n, _ = uv.shape
    ct, st = torch.cos(th), torch.sin(th)
    a2, b2 = a * a, b * b
    hu = torch.sqrt(a2 * ct * ct + b2 * st * st) / a
    hv = torch.sqrt(a2 * st * st + b2 * ct * ct) / b
    R = torch.sqrt(hu * hu + hv * hv)
    scale = (R.unsqueeze(2) + R.unsqueeze(1)) ** 2
    rank = _pdist2_t(uv) / scale.clamp_min(1e-12)
    nbr = rank.topk(k, dim=-1, largest=False).indices
    pi_ = torch.arange(n, device=uv.device).view(1, n, 1).expand(B, n, k)
    return pi_.reshape(B, n * k), nbr.reshape(B, n * k)


def _gather_pairs_t(t, idx):
    return t[:, idx] if idx.dim() == 1 else t.gather(1, idx)


def _violations_t(uv, th, Wu, Hv, a, b, pi_, pj_, coarse, refine):
    ct, st = torch.cos(th), torch.sin(th)
    a2, b2 = a * a, b * b
    A11 = a2 * ct * ct + b2 * st * st
    A22 = a2 * st * st + b2 * ct * ct
    A12 = (a2 - b2) * ct * st
    hu, hv = torch.sqrt(A11) / a, torch.sqrt(A22) / b
    u, v = uv[..., 0], uv[..., 1]
    wall = torch.stack([hu - u, u + hu - Wu.unsqueeze(-1),
                        hv - v, v + hv - Hv.unsqueeze(-1)], dim=-1)

    rx = (_gather_pairs_t(u, pj_) - _gather_pairs_t(u, pi_)) * a
    ry = (_gather_pairs_t(v, pj_) - _gather_pairs_t(v, pi_)) * b
    Ai = (_gather_pairs_t(A11, pi_), _gather_pairs_t(A12, pi_), _gather_pairs_t(A22, pi_))
    Aj = (_gather_pairs_t(A11, pj_), _gather_pairs_t(A12, pj_), _gather_pairs_t(A22, pj_))
    with torch.no_grad():
        lam = _lambda_star_t(Ai, Aj, rx, ry, coarse, refine)
    mu2 = _pw_F_t(lam.unsqueeze(-1), Ai, Aj, rx, ry).squeeze(-1)
    return wall, torch.sqrt(mu2.clamp_min(1e-12))


def _lbfgs_direction(g, hist_s, hist_y, hist_valid, H_diag):
    reduce_dims = tuple(range(1, g.dim()))
    bshape = (-1,) + (1,) * (g.dim() - 1)
    H = hist_s.shape[1]
    q = g
    alpha = []
    for i in range(H - 1, -1, -1):
        s_i, y_i, v_i = hist_s[:, i], hist_y[:, i], hist_valid[:, i]
        sy_i = (s_i * y_i).sum(dim=reduce_dims).clamp_min(1e-30)
        a_i = (s_i * q).sum(dim=reduce_dims) / sy_i
        q = q - torch.where(v_i.view(*bshape), y_i * a_i.view(*bshape), torch.zeros_like(y_i))
        alpha.append(a_i)
    alpha.reverse()
    r = q * H_diag.view(*bshape)
    for i in range(H):
        s_i, y_i, v_i = hist_s[:, i], hist_y[:, i], hist_valid[:, i]
        sy_i = (s_i * y_i).sum(dim=reduce_dims).clamp_min(1e-30)
        b_i = (y_i * r).sum(dim=reduce_dims) / sy_i
        upd = s_i * (alpha[i] - b_i).view(*bshape)
        r = r + torch.where(v_i.view(*bshape), upd, torch.zeros_like(upd))
    return -r


def _trust_region_step(d, max_step):
    reduce_dims = tuple(range(1, d.dim()))
    bshape = (-1,) + (1,) * (d.dim() - 1)
    max_abs = d.abs().amax(dim=reduce_dims).clamp_min(1e-12)
    scale = (max_step / max_abs).clamp(max=1.0)
    return d * scale.view(*bshape)


def _repair_t(uv0, th0, Wu0, Hv0, a, b, pi_full, pj_full, steps, lr_pos, lr_th, coarse, refine,
              shape_bounds, knn_every=KNN_EVERY, pw=None, check_every=25, lr_cell=0.0,
              free_mask=None):
    n = uv0.shape[1]
    B = uv0.shape[0]
    dtype, dev = uv0.dtype, uv0.device
    uv = uv0.detach().clone().requires_grad_(True)
    th = th0.detach().clone().requires_grad_(True)
    cell = lr_cell > 0.0
    if cell:
        Wu = Wu0.detach().clone().requires_grad_(True)
        Hv = Hv0.detach().clone().requires_grad_(True)
    else:
        Wu, Hv = Wu0, Hv0

    hist_s_uv = torch.zeros(B, LBFGS_HISTORY, n, 2, dtype=dtype, device=dev)
    hist_y_uv = torch.zeros(B, LBFGS_HISTORY, n, 2, dtype=dtype, device=dev)
    hist_valid_uv = torch.zeros(B, LBFGS_HISTORY, dtype=torch.bool, device=dev)
    H_diag_uv = torch.ones(B, dtype=dtype, device=dev)
    hist_s_th = torch.zeros(B, LBFGS_HISTORY, n, dtype=dtype, device=dev)
    hist_y_th = torch.zeros(B, LBFGS_HISTORY, n, dtype=dtype, device=dev)
    hist_valid_th = torch.zeros(B, LBFGS_HISTORY, dtype=torch.bool, device=dev)
    H_diag_th = torch.ones(B, dtype=dtype, device=dev)
    prev_uv, prev_th = torch.empty_like(uv), torch.empty_like(th)
    prev_guv, prev_gth = torch.empty_like(uv), torch.empty_like(th)
    if cell:
        alpha_Wu = torch.full((B,), lr_cell, dtype=dtype, device=dev)
        alpha_Hv = torch.full((B,), lr_cell, dtype=dtype, device=dev)
        prev_Wu, prev_Hv = torch.empty_like(Wu), torch.empty_like(Hv)
        prev_gWu, prev_gHv = torch.empty_like(Wu), torch.empty_like(Hv)
    have_prev = False
    seg_pairs = None

    for step in range(steps):
        if step % knn_every == 0:
            with torch.no_grad():
                seg_pairs = [(lo, hi, *_knn_pairs_t(uv[lo:hi].detach(), th[lo:hi].detach(),
                                                    a[lo:hi], b[lo:hi], k))
                            for lo, hi, k in shape_bounds]
        losses, seg_info = [], []
        for lo, hi, pi_seg, pj_seg in seg_pairs:
            wall_seg, mu_seg = _violations_t(uv[lo:hi], th[lo:hi], Wu[lo:hi], Hv[lo:hi],
                                             a[lo:hi], b[lo:hi], pi_seg, pj_seg, coarse, refine)
            wv = torch.clamp_min(wall_seg + WALL_TARGET, 0.0)
            pv = torch.clamp_min(1.0 + MU_TARGET - mu_seg, 0.0)
            if pw is not None:
                w_idx_seg = torch.minimum(pi_seg, pj_seg) * n + torch.maximum(pi_seg, pj_seg)
                wgt_seg = pw[lo:hi].gather(1, w_idx_seg)
                pv2 = wgt_seg * pv * pv
            else:
                w_idx_seg = None
                pv2 = pv * pv
            loss_seg = (wv * wv).sum(dim=(1, 2)) + PAIR_W * pv2.sum(dim=1)
            losses.append(loss_seg)
            seg_info.append((lo, hi, pi_seg, pj_seg, w_idx_seg, mu_seg))
        feas_loss = torch.cat(losses, dim=0)
        loss = feas_loss + AREA_SCALE * (Wu * Hv) if cell else feas_loss
        params = [uv, th] + ([Wu, Hv] if cell else [])
        grads = torch.autograd.grad(loss.sum(), params)
        guv, gth = grads[0], grads[1]
        if cell:
            gWu, gHv = grads[2], grads[3]
        with torch.no_grad():
            guv = torch.nan_to_num(guv, nan=0.0, posinf=0.0, neginf=0.0)
            gth = torch.nan_to_num(gth, nan=0.0, posinf=0.0, neginf=0.0)
            if cell:
                gWu = torch.nan_to_num(gWu, nan=0.0, posinf=0.0, neginf=0.0)
                gHv = torch.nan_to_num(gHv, nan=0.0, posinf=0.0, neginf=0.0)
            if free_mask is not None:
                guv = guv * free_mask.unsqueeze(-1)
                gth = gth * free_mask

            first_step = not have_prev
            if have_prev:
                s_uv, y_uv = uv - prev_uv, guv - prev_guv
                sy_uv = (s_uv * y_uv).sum(dim=(1, 2))
                valid_uv = (sy_uv > 1e-12) & torch.isfinite(sy_uv)
                hist_s_uv = torch.cat([hist_s_uv[:, 1:], s_uv.unsqueeze(1)], dim=1)
                hist_y_uv = torch.cat([hist_y_uv[:, 1:], y_uv.unsqueeze(1)], dim=1)
                hist_valid_uv = torch.cat([hist_valid_uv[:, 1:], valid_uv.unsqueeze(1)], dim=1)
                yy_uv = (y_uv * y_uv).sum(dim=(1, 2)).clamp_min(1e-30)
                H_new_uv = (sy_uv / yy_uv).clamp(BB_LO, BB_HI)
                H_diag_uv = torch.where(valid_uv, H_new_uv, H_diag_uv)

                s_th, y_th = th - prev_th, gth - prev_gth
                sy_th = (s_th * y_th).sum(dim=1)
                valid_th = (sy_th > 1e-12) & torch.isfinite(sy_th)
                hist_s_th = torch.cat([hist_s_th[:, 1:], s_th.unsqueeze(1)], dim=1)
                hist_y_th = torch.cat([hist_y_th[:, 1:], y_th.unsqueeze(1)], dim=1)
                hist_valid_th = torch.cat([hist_valid_th[:, 1:], valid_th.unsqueeze(1)], dim=1)
                yy_th = (y_th * y_th).sum(dim=1).clamp_min(1e-30)
                H_new_th = (sy_th / yy_th).clamp(BB_LO, BB_HI)
                H_diag_th = torch.where(valid_th, H_new_th, H_diag_th)

                if cell:
                    s_Wu, y_Wu = Wu - prev_Wu, gWu - prev_gWu
                    sy_Wu = s_Wu * y_Wu
                    cand_Wu = (s_Wu * s_Wu / sy_Wu.clamp_min(1e-30)) \
                        .clamp(BB_LO * lr_cell, BB_HI * lr_cell)
                    use_Wu = (sy_Wu > 1e-12) & torch.isfinite(cand_Wu)
                    alpha_Wu = torch.where(use_Wu, cand_Wu, alpha_Wu)

                    s_Hv, y_Hv = Hv - prev_Hv, gHv - prev_gHv
                    sy_Hv = s_Hv * y_Hv
                    cand_Hv = (s_Hv * s_Hv / sy_Hv.clamp_min(1e-30)) \
                        .clamp(BB_LO * lr_cell, BB_HI * lr_cell)
                    use_Hv = (sy_Hv > 1e-12) & torch.isfinite(cand_Hv)
                    alpha_Hv = torch.where(use_Hv, cand_Hv, alpha_Hv)
            prev_uv.copy_(uv); prev_th.copy_(th)
            prev_guv.copy_(guv); prev_gth.copy_(gth)
            if cell:
                prev_Wu.copy_(Wu); prev_Hv.copy_(Hv)
                prev_gWu.copy_(gWu); prev_gHv.copy_(gHv)
            have_prev = True

            if first_step:
                d_uv, d_th = -guv, -gth
            else:
                d_uv = _lbfgs_direction(guv, hist_s_uv, hist_y_uv, hist_valid_uv, H_diag_uv)
                d_th = _lbfgs_direction(gth, hist_s_th, hist_y_th, hist_valid_th, H_diag_th)
            step_uv = _trust_region_step(d_uv, lr_pos)
            step_th = _trust_region_step(d_th, lr_th)
            uv = (uv + step_uv).detach().requires_grad_(True)
            th = (th + step_th).detach().requires_grad_(True)
            if cell:
                Wu = (Wu - alpha_Wu * gWu).clamp_min(1e-3).detach().requires_grad_(True)
                Hv = (Hv - alpha_Hv * gHv).clamp_min(1e-3).detach().requires_grad_(True)
        if (step + 1) % check_every == 0 and bool((feas_loss <= 0.0).all()):
            break

    if cell:
        Wu, Hv = Wu.detach(), Hv.detach()
    with torch.no_grad():
        wall, mu_all = _violations_t(uv, th, Wu, Hv, a, b, pi_full, pj_full, coarse, refine)
        wmax, mumin = wall.amax(dim=(1, 2)), mu_all.amin(dim=1)
        ok = (wmax <= -WALL_ACCEPT) & (mumin >= 1.0 + MU_ACCEPT)
        k_max = max(k for _, _, k in shape_bounds)
        miss_parts, w_idx_parts, mu_c_parts = [], [], []
        for lo, hi, pi_seg, pj_seg, w_idx_seg, _ in seg_info:
            _, mu_c_seg = _violations_t(uv[lo:hi], th[lo:hi], Wu[lo:hi], Hv[lo:hi],
                                        a[lo:hi], b[lo:hi], pi_seg, pj_seg, coarse, refine)
            miss_parts.append((mumin[lo:hi] < mu_c_seg.amin(dim=1) - 1e-6).float())
            if w_idx_seg is not None:
                pad = k_max * n - w_idx_seg.shape[1]
                if pad > 0:
                    w_idx_seg = torch.cat([w_idx_seg, torch.zeros(hi - lo, pad,
                                          dtype=w_idx_seg.dtype, device=dev)], dim=1)
                    mu_c_seg = torch.cat([mu_c_seg, torch.full((hi - lo, pad), 2.0,
                                          dtype=mu_c_seg.dtype, device=dev)], dim=1)
                w_idx_parts.append(w_idx_seg)
                mu_c_parts.append(mu_c_seg)
        miss = torch.cat(miss_parts, dim=0)
        w_idx = torch.cat(w_idx_parts, dim=0) if w_idx_parts else None
        mu_c = torch.cat(mu_c_parts, dim=0) if mu_c_parts else None
    info = {"wmax": wmax, "mumin": mumin, "w_idx": w_idx, "mu_c": mu_c, "miss": miss}
    return uv.detach(), th.detach(), Wu, Hv, ok, info


def _relocate_t(uv, th, Wu, Hv, a, b, n_cand):
    B, n, _ = uv.shape
    ct, st = torch.cos(th), torch.sin(th)
    a2, b2 = a * a, b * b
    hu = torch.sqrt(a2 * ct * ct + b2 * st * st) / a
    hv = torch.sqrt(a2 * st * st + b2 * ct * ct) / b

    d2 = _pdist2_t(uv)
    tight = d2.amin(dim=2).topk(max(2, n // 8), dim=1, largest=False).indices
    pick = torch.randint(tight.shape[1], (B, 1), device=uv.device)
    m = tight.gather(1, pick)

    hu_m, hv_m = hu.gather(1, m), hv.gather(1, m)
    span_u = (Wu.view(B, 1) - 2.0 * hu_m).clamp_min(0.0)
    span_v = (Hv.view(B, 1) - 2.0 * hv_m).clamp_min(0.0)
    cu = hu_m + torch.rand(B, n_cand, device=uv.device, dtype=uv.dtype) * span_u
    cv = hv_m + torch.rand(B, n_cand, device=uv.device, dtype=uv.dtype) * span_v
    cand = torch.stack([cu, cv], dim=-1)

    diff = cand.unsqueeze(2) - uv.unsqueeze(1)
    d = (diff * diff).sum(-1)
    d.scatter_(2, m.view(B, 1, 1).expand(B, n_cand, 1), 1e9)
    best = d.amin(dim=2).argmax(dim=1).view(B, 1, 1)
    newpos = cand.gather(1, best.expand(B, 1, 2))
    newth = torch.rand(B, 1, device=uv.device, dtype=uv.dtype) * math.pi
    return (uv.scatter(1, m.view(B, 1, 1).expand(B, 1, 2), newpos),
            th.scatter(1, m, newth))


def _sa_relocate_t(uv, th, Wu, Hv, a, b, pw, n_cand, m_move):
    B, n, _ = uv.shape
    tri_mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=uv.device), diagonal=1)
    weighted = pw.view(B, n, n) * tri_mask
    trouble = weighted.sum(dim=2) + weighted.sum(dim=1)
    target = trouble.argmax(dim=1, keepdim=True)

    d2 = _pdist2_t(uv)
    chosen = torch.zeros(B, n, dtype=torch.bool, device=uv.device)
    chosen.scatter_(1, target, True)
    frontier = target
    movers = [target]
    for _ in range(m_move - 1):
        cand_d = d2.gather(1, frontier.unsqueeze(-1).expand(B, 1, n)).squeeze(1)
        cand_d = cand_d * (0.85 + 0.3 * torch.rand(B, n, device=uv.device, dtype=uv.dtype))
        cand_d = cand_d.masked_fill(chosen, float("inf"))
        nxt = cand_d.argmin(dim=1, keepdim=True)
        chosen.scatter_(1, nxt, True)
        movers.append(nxt)
        frontier = torch.where(torch.rand(B, 1, device=uv.device) < 0.5, nxt, target)
    movers = torch.cat(movers, dim=1)

    ct, st = torch.cos(th), torch.sin(th)
    a2, b2 = a * a, b * b
    hu = torch.sqrt(a2 * ct * ct + b2 * st * st) / a
    hv = torch.sqrt(a2 * st * st + b2 * ct * ct) / b

    uv, th = uv.clone(), th.clone()
    target_pos = uv.gather(1, target.unsqueeze(-1).expand(B, 1, 2)).squeeze(1)
    uv = _loosen_t(uv, target_pos, Wu, Hv, LOOSEN_STRENGTH)
    for j in range(m_move):
        m = movers[:, j:j + 1]
        hu_m, hv_m = hu.gather(1, m), hv.gather(1, m)
        span_u = (Wu.view(B, 1) - 2.0 * hu_m).clamp_min(0.0)
        span_v = (Hv.view(B, 1) - 2.0 * hv_m).clamp_min(0.0)
        cu = hu_m + torch.rand(B, n_cand, device=uv.device, dtype=uv.dtype) * span_u
        cv = hv_m + torch.rand(B, n_cand, device=uv.device, dtype=uv.dtype) * span_v
        cand = torch.stack([cu, cv], dim=-1)
        diff = cand.unsqueeze(2) - uv.unsqueeze(1)
        d = (diff * diff).sum(-1)
        d.scatter_(2, m.view(B, 1, 1).expand(B, n_cand, 1), 1e9)
        best = d.amin(dim=2).argmax(dim=1).view(B, 1, 1)
        newpos = cand.gather(1, best.expand(B, 1, 2))
        newth = torch.rand(B, 1, device=uv.device, dtype=uv.dtype) * math.pi
        uv = uv.scatter(1, m.view(B, 1, 1).expand(B, 1, 2), newpos)
        th = th.scatter(1, m, newth)
    return uv, th


def _group_mask_t(uv, group_size, centres=None):
    B, n, _ = uv.shape
    dev, dtype = uv.device, uv.dtype
    if centres is None:
        lo = uv.amin(dim=1)
        hi = uv.amax(dim=1)
        centres = lo + (hi - lo) * torch.rand(B, 2, device=dev, dtype=dtype)
    d2 = ((uv - centres.unsqueeze(1)) ** 2).sum(-1)
    d2 = d2 * (0.9 + 0.2 * torch.rand(B, n, device=dev, dtype=dtype))
    k = max(1, min(n, group_size))
    idx = d2.topk(k, dim=1, largest=False).indices
    mask = torch.zeros(B, n, device=dev, dtype=dtype)
    mask.scatter_(1, idx, 1.0)
    return mask


def _jam_mask_t(uv, th, Wu, Hv, a, b, k_free):
    B, n, _ = uv.shape
    dev, dtype = uv.device, uv.dtype
    ct, st = torch.cos(th), torch.sin(th)
    a2, b2 = a * a, b * b
    hu = torch.sqrt(a2 * ct * ct + b2 * st * st) / a
    hv = torch.sqrt(a2 * st * st + b2 * ct * ct) / b
    du = torch.minimum(uv[:, :, 0], Wu.view(B, 1) - uv[:, :, 0]) - hu
    dv = torch.minimum(uv[:, :, 1], Hv.view(B, 1) - uv[:, :, 1]) - hv
    at_vertical = du <= dv
    gap = torch.where(at_vertical, du, dv)
    hx = torch.sqrt(a2 * ct * ct + b2 * st * st)
    hy = torch.sqrt(a2 * st * st + b2 * ct * ct)
    h_perp = torch.where(at_vertical, hx, hy)
    touching = gap <= WALL_TOUCH_SLACK
    waste = torch.where(touching, h_perp - b, torch.full_like(h_perp, -1e9))
    target = waste.argmax(dim=1, keepdim=True)

    d2 = _pdist2_t(uv)
    d2_t = d2.gather(1, target.unsqueeze(-1).expand(B, 1, n)).squeeze(1)
    d2_t.scatter_(1, target, -1.0)
    idx = d2_t.argsort(dim=1)[:, :max(1, k_free)]
    mask = torch.zeros(B, n, device=dev, dtype=dtype)
    mask.scatter_(1, idx, 1.0)
    tmask = torch.zeros(B, n, device=dev, dtype=dtype)
    tmask.scatter_(1, target, 1.0)
    centre = uv.gather(1, target.unsqueeze(-1).expand(B, 1, 2)).squeeze(1)
    return mask, tmask, centre


def _boundary_mask_t(uv, th, Wu, Hv, a, b, depth):
    B, n, _ = uv.shape
    ct, st = torch.cos(th), torch.sin(th)
    a2, b2 = a * a, b * b
    hu = torch.sqrt(a2 * ct * ct + b2 * st * st) / a
    hv = torch.sqrt(a2 * st * st + b2 * ct * ct) / b
    du = torch.minimum(uv[:, :, 0], Wu.view(B, 1) - uv[:, :, 0]) / hu.clamp_min(1e-9)
    dv = torch.minimum(uv[:, :, 1], Hv.view(B, 1) - uv[:, :, 1]) / hv.clamp_min(1e-9)
    return (torch.minimum(du, dv) <= depth).to(uv.dtype)


def _rattler_mask_t(uv, th, Wu, Hv, a, b, pi_full, pj_full, coarse, refine, k_free):
    B, n, _ = uv.shape
    dev, dtype = uv.device, uv.dtype
    with torch.no_grad():
        _, mu_full = _violations_t(uv, th, Wu, Hv, a, b, pi_full, pj_full, coarse, refine)
        dense = torch.full((B, n, n), 9.0, device=dev, dtype=dtype)
        dense[:, pi_full, pj_full] = mu_full
        dense[:, pj_full, pi_full] = mu_full
        contacts = (dense < CONTACT_MU).sum(dim=2)
    loose = contacts.float().topk(max(1, k_free), dim=1, largest=False).indices
    d2 = _pdist2_t(uv)
    mask = torch.zeros(B, n, device=dev, dtype=dtype)
    mask.scatter_(1, loose, 1.0)
    shell = d2.gather(1, loose.unsqueeze(-1).expand(-1, -1, n))
    nbr = shell.topk(min(n, RATTLER_SHELL + 1), dim=2, largest=False).indices
    mask.scatter_(1, nbr.reshape(B, -1), 1.0)
    return mask


def _symmetrise_t(uv, th, partner):
    centroid = uv.mean(dim=1, keepdim=True)
    mirrored = 2.0 * centroid - uv
    part_uv = mirrored.index_select(1, partner)
    sym_uv = 0.5 * (uv + part_uv)

    part_th = th.index_select(1, partner)
    c = torch.cos(2.0 * th) + torch.cos(2.0 * part_th)
    sv = torch.sin(2.0 * th) + torch.sin(2.0 * part_th)
    sym_th = 0.5 * torch.atan2(sv, c)
    return sym_uv, sym_th


def _loosen_t(uv, target, Wu, Hv, strength):
    d = uv - target.unsqueeze(1)
    r2 = (d * d).sum(-1, keepdim=True)
    span = (Wu * Wu + Hv * Hv).view(-1, 1, 1)
    push = strength * torch.exp(-4.0 * r2 / span)
    return uv + d * push


def _crossover_t(best_uv, best_th, best_Wu, best_Hv, pa, pb, rng):
    n = best_uv.shape[1]
    m = int(rng.integers(n // 3, 2 * n // 3 + 1))
    ua, ta = best_uv[pa], best_th[pa]
    ub, tb = best_uv[pb], best_th[pb]
    ia = torch.argsort(ua[:, 0])[:m]
    ib = torch.argsort(ub[:, 0], descending=True)[:n - m]
    uv = torch.cat([ua[ia], ub[ib]], dim=0)
    th = torch.cat([ta[ia], tb[ib]], dim=0)
    Wu = max(float(best_Wu[pa]), float(best_Wu[pb])) * CROSSOVER_INFLATE
    Hv = max(float(best_Hv[pa]), float(best_Hv[pb])) * CROSSOVER_INFLATE
    return uv, th, Wu, Hv


def _initial_population_t(keys, R_list, rng, warm, slack=1.002):
    uv, th, Wu, Hv = [], [], [], []
    for key, replicas in zip(keys, R_list):
        u, t, W, H = make_population(replicas, rng, slack=slack)
        if warm and key in warm and replicas > 1:
            w = warm[key]
            u[1], t[1] = np.asarray(w["uv"]), np.asarray(w["theta"])
            W[1], H[1] = w["Wu"] * WARM_SLACK, w["Hv"] * WARM_SLACK
            u[1] = u[1] * WARM_SLACK
            for i in range(2, min(replicas, 2 + max(1, replicas // 4))):
                u[i] = np.asarray(w["uv"]) * 1.02 + rng.normal(scale=0.02, size=(N_ELLIPSES, 2))
                t[i] = np.asarray(w["theta"]) + rng.normal(scale=0.06, size=N_ELLIPSES)
                W[i], H[i] = w["Wu"] * 1.02, w["Hv"] * 1.02
        uv.append(u)
        th.append(t)
        Wu.append(W)
        Hv.append(H)
    return (np.concatenate(uv), np.concatenate(th),
            np.concatenate(Wu), np.concatenate(Hv))


def _bounds_tensor(keys, bounds, tune, name, default, dtype, device):
    out = torch.empty(bounds[-1][1], dtype=dtype, device=device)
    for key, (lo, hi) in zip(keys, bounds):
        out[lo:hi] = tune.get(key, {}).get(name, default)
    return out


def _csv(vals, spec=""):
    return ",".join(format(v, spec) for v in vals)


def _adaptive_fan(champ_aspect, rng):
    if rng.random() < ADAPTIVE_FAN_EPSILON or not math.isfinite(champ_aspect) or champ_aspect <= 0:
        return ASPECT_FAN
    lo = max(ASPECT_FAN[0], champ_aspect / ADAPTIVE_FAN_WIDTH)
    hi = min(ASPECT_FAN[1], champ_aspect * ADAPTIVE_FAN_WIDTH)
    return (lo, hi) if lo < hi else ASPECT_FAN


def search(keys, seconds, replicas_per_shape, repair_steps=120, device=None,
           dtype=torch.float32, seed=0, coarse=13, refine=6, lr_pos=0.2, lr_th=0.4,
           lr_cell=LR_CELL, warm=None, escape_every=6, reseed_every=60, on_checkpoint=None,
           force_branches=False, checkpoint_every=180.0, log_every=60.0, verbose=True, tag="",
           migrate=None, migrate_every=10):
    dev = torch.device(device) if device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    keys = list(keys)
    S = len(keys)
    if isinstance(replicas_per_shape, dict):
        R_list = [replicas_per_shape[k] for k in keys]
    else:
        R_list = [replicas_per_shape] * S
    bounds = []
    lo = 0
    for R in R_list:
        bounds.append((lo, lo + R))
        lo += R
    reserve_lo = [hi - int((hi - lo_) * RESERVE_FRAC) if (hi - lo_) >= 8 else hi
                  for lo_, hi in bounds]
    B, n = lo, N_ELLIPSES
    partner_t = torch.tensor(_point_pairs(hex67()[0]), dtype=torch.long, device=dev)
    cull_k_list = [SHAPE_TUNE.get(k, {}).get("cull_k", CULL_K) for k in keys]
    group_of = {k: gi for gi, group in enumerate(CULL_GROUPS) for k in group}
    shape_bounds = []
    si = 0
    while si < S:
        gi = group_of.get(keys[si], -si - 1)
        sj = si
        while sj + 1 < S and group_of.get(keys[sj + 1], -(sj + 1) - 1) == gi:
            sj += 1
        shape_bounds.append((bounds[si][0], bounds[sj][1], max(cull_k_list[si:sj + 1])))
        si = sj + 1
    t0 = time.time()
    stall_escape_b = _bounds_tensor(keys, bounds, SHAPE_TUNE, "stall_escape", STALL_ESCAPE,
                                    torch.int32, dev)
    stall_reset_b = _bounds_tensor(keys, bounds, SHAPE_TUNE, "stall_reset", STALL_RESET,
                                   torch.int32, dev)
    gls_max_b = _bounds_tensor(keys, bounds, SHAPE_TUNE, "gls_max", GLS_MAX,
                               dtype, dev).unsqueeze(1)

    pre = f"  [{tag}] " if tag else "  "
    uv_np, th_np, Wu_np, Hv_np = _initial_population_t(keys, R_list, rng, warm)
    ab = np.array([SHAPE_AB[k] for k in keys])
    a = torch.tensor(np.repeat(ab[:, 0], R_list), dtype=dtype, device=dev).unsqueeze(1)
    b = torch.tensor(np.repeat(ab[:, 1], R_list), dtype=dtype, device=dev).unsqueeze(1)
    uv = torch.tensor(uv_np, dtype=dtype, device=dev)
    th = torch.tensor(th_np, dtype=dtype, device=dev)
    Wu = torch.tensor(Wu_np, dtype=dtype, device=dev)
    Hv = torch.tensor(Hv_np, dtype=dtype, device=dev)
    ii, jj = np.triu_indices(n, 1)
    pi_full = torch.tensor(ii, dtype=torch.long, device=dev)
    pj_full = torch.tensor(jj, dtype=torch.long, device=dev)
    pw = torch.ones(B, n * n, dtype=dtype, device=dev)
    rep_kw = dict(a=a, b=b, pi_full=pi_full, pj_full=pj_full, lr_pos=lr_pos, lr_th=lr_th,
                  coarse=coarse, refine=refine, shape_bounds=shape_bounds, pw=pw, lr_cell=lr_cell)
    rep_kw_bnd = dict(rep_kw, lr_pos=lr_pos * BOUNDARY_LR_POS_MULT,
                      lr_th=lr_th * BOUNDARY_LR_TH_MULT)

    ok = torch.zeros(B, dtype=torch.bool, device=dev)
    for _ in range(12):
        uv, th, Wu, Hv, ok, _ = _repair_t(uv, th, Wu, Hv, steps=max(200, repair_steps), **rep_kw)
        if bool(ok.all()):
            break
        grow = torch.where(ok, torch.ones_like(Wu), torch.full_like(Wu, 1.03))
        uv, Wu, Hv = uv * grow.view(B, 1, 1), Wu * grow, Hv * grow
    if force_branches:
        ok[0] = False
    if not bool(ok.all()):
        h_uv, h_W, h_H = hex67()
        fb = torch.tensor(h_uv * 1.002, dtype=dtype, device=dev).unsqueeze(0)
        uv = torch.where((~ok).view(B, 1, 1), fb, uv)
        th = torch.where((~ok).view(B, 1), torch.zeros_like(th), th)
        Wu = torch.where(~ok, torch.full_like(Wu, h_W * 1.002), Wu)
        Hv = torch.where(~ok, torch.full_like(Hv, h_H * 1.002), Hv)
        if verbose:
            print(f"{pre}initial repair: {int((~ok).sum())}/{B} replicas "
                  f"fell back to pristine hex")

    best_uv, best_th = uv.clone(), th.clone()
    best_Wu, best_Hv = Wu.clone(), Hv.clone()
    best_area = Wu * Hv
    gamma = torch.full((B,), GAMMA0, dtype=dtype, device=dev)
    stall = torch.zeros(B, dtype=torch.int32, device=dev)

    def extract():
        out = {}
        for si, key in enumerate(keys):
            seg_lo, seg_hi = bounds[si]
            seg = best_area[seg_lo:seg_hi]
            j = seg_lo + int(torch.argmin(seg))
            out[key] = {"uv": best_uv[j].double().cpu().numpy(),
                        "theta": best_th[j].double().cpu().numpy(),
                        "Wu": float(best_Wu[j]), "Hv": float(best_Hv[j]),
                        "area": float(best_area[j])}
        return out

    def implied(area):
        return (N_ELLIPSES * math.pi / area * 1000.0) ** 2

    outer = 0
    n_immigrants, n_escape, n_reset, n_cross, n_sa = [0] * S, [0] * S, [0] * S, [0] * S, [0] * S
    n_deep = [0] * S
    n_sym = [0] * S
    n_bnd = [0] * S
    n_jam = [0] * S
    miss_ema = [0.0] * S
    next_log, next_ck = t0 + log_every, t0 + checkpoint_every
    while time.time() - t0 < seconds:
        outer += 1
        shrink_per_axis = 1.0 - torch.sqrt(1.0 - gamma)
        jit_amp = (ASPECT_JITTER_REL * shrink_per_axis).clamp_max(ASPECT_JITTER_MAX)
        jit = (torch.empty(B, dtype=dtype, device=dev).uniform_(-1.0, 1.0) * jit_amp).exp()
        sx = torch.sqrt(1.0 - gamma) * jit
        sy = (1.0 - gamma) / sx
        Wn, Hn = Wu * sx, Hv * sy
        uvn, thn, Wun, Hvn, ok, info = _repair_t(
            uv * torch.stack([sx, sy], dim=1).unsqueeze(1), th, Wn, Hn, steps=repair_steps,
            **rep_kw)
        uv = torch.where(ok.view(B, 1, 1), uvn, uv)
        th = torch.where(ok.view(B, 1), thn, th)
        Wu, Hv = torch.where(ok, Wun, Wu), torch.where(ok, Hvn, Hv)
        gamma = torch.where(ok, gamma * GAMMA_UP, gamma * GAMMA_DOWN).clamp(GAMMA_MIN, GAMMA_MAX)
        for si, (seg_lo, seg_hi) in enumerate(bounds):
            miss_ema[si] = 0.9 * miss_ema[si] + 0.1 * float(info["miss"][seg_lo:seg_hi].mean())

        pw.mul_(GLS_DECAY).add_(1.0 - GLS_DECAY)
        if info["w_idx"] is not None:
            blocked = (info["mu_c"] < 1.0 + MU_ACCEPT) & (~ok).unsqueeze(1)
            cur = pw.gather(1, info["w_idx"])
            pw.scatter_(1, info["w_idx"],
                        torch.where(blocked, torch.minimum(cur * GLS_UP, gls_max_b), cur))

        area = Wu * Hv
        imp = area < best_area
        best_area = torch.where(imp, area, best_area)
        best_uv = torch.where(imp.view(B, 1, 1), uv, best_uv)
        best_th = torch.where(imp.view(B, 1), th, best_th)
        best_Wu, best_Hv = torch.where(imp, Wu, best_Wu), torch.where(imp, Hv, best_Hv)
        stall = torch.where(imp, torch.zeros_like(stall), stall + 1)

        if outer % escape_every == 0:
            sel = torch.ones_like(stall, dtype=torch.bool) if force_branches \
                else stall >= stall_escape_b
            if bool(sel.any()):
                eu, et = _relocate_t(uv, th, Wu, Hv, a, b, RELOCATE_CAND)
                sig = float(rng.uniform(0.01, 0.05))
                eu = eu + torch.randn_like(eu) * sig
                et = et + torch.randn_like(et) * (sig * 4.0)
                eW, eH = Wu * KICK, Hv * KICK
                eu, et, eWu, eHv, ok2, _ = _repair_t(eu, et, eW, eH, steps=repair_steps, **rep_kw)
                take = sel & ok2
                uv = torch.where(take.view(B, 1, 1), eu, uv)
                th = torch.where(take.view(B, 1), et, th)
                Wu, Hv = torch.where(take, eWu, Wu), torch.where(take, eHv, Hv)
                gamma = torch.where(take, torch.full_like(gamma, GAMMA0), gamma)
                stall = torch.where(take, torch.full_like(stall, -ESCAPE_GRACE), stall)
                for si, (seg_lo, seg_hi) in enumerate(bounds):
                    n_escape[si] += int(take[seg_lo:seg_hi].sum())

        if outer % (SA_EVERY_MULT * escape_every) == 0:
            if force_branches:
                sel_sa = torch.ones_like(stall, dtype=torch.bool)
            else:
                sel_sa = torch.zeros(B, dtype=torch.bool, device=dev)
                for si, (seg_lo, seg_hi) in enumerate(bounds):
                    pool_hi = reserve_lo[si]
                    if pool_hi - seg_lo < 2:
                        continue
                    seg_area = best_area[seg_lo:pool_hi]
                    finite = torch.isfinite(seg_area)
                    if not bool(finite.any()):
                        continue
                    ranked = torch.where(finite, seg_area, torch.full_like(seg_area, -1.0))
                    seg_order = torch.argsort(ranked, descending=True)
                    n_pick = min(max(1, int((pool_hi - seg_lo) * SA_TARGET_FRAC)),
                                 int(finite.sum()))
                    sel_sa[seg_lo + seg_order[:n_pick]] = True
            if bool(sel_sa.any()):
                m_move = int(rng.integers(1, SA_REMOVE_K_MAX + 1))
                cur_uv, cur_th = _sa_relocate_t(uv, th, Wu, Hv, a, b, pw, RELOCATE_CAND, m_move)
                cur_Wu, cur_Hv = Wu * KICK, Hv * KICK
                cur_area = cur_Wu * cur_Hv
                cur_ok = torch.zeros(B, dtype=torch.bool, device=dev)
                for r in range(SA_ROUNDS):
                    p = r / max(SA_ROUNDS - 1, 1)
                    temp = SA_T_HIGH ** (1.0 - p) * SA_T_LOW ** p
                    sig = float(rng.uniform(*SA_KICK_RANGE))
                    ku = cur_uv + torch.randn_like(cur_uv) * sig
                    kt = cur_th + torch.randn_like(cur_th) * (sig * 4.0)
                    ru, rt, rWu, rHv, rok, _ = _repair_t(
                        ku, kt, cur_Wu, cur_Hv, steps=max(20, repair_steps // 4), **rep_kw)
                    rarea = rWu * rHv
                    delta = rarea - cur_area
                    u01 = torch.rand(B, device=dev, dtype=dtype)
                    metropolis = u01 < torch.exp(-delta.clamp_min(0.0) / temp)
                    accept = rok & (~cur_ok | (delta <= 0.0) | metropolis)
                    cur_uv = torch.where(accept.view(B, 1, 1), ru, cur_uv)
                    cur_th = torch.where(accept.view(B, 1), rt, cur_th)
                    cur_Wu, cur_Hv = torch.where(accept, rWu, cur_Wu), torch.where(accept, rHv, cur_Hv)
                    cur_area = torch.where(accept, rarea, cur_area)
                    cur_ok = cur_ok | accept
                take_sa = sel_sa & cur_ok
                uv = torch.where(take_sa.view(B, 1, 1), cur_uv, uv)
                th = torch.where(take_sa.view(B, 1), cur_th, th)
                Wu, Hv = torch.where(take_sa, cur_Wu, Wu), torch.where(take_sa, cur_Hv, Hv)
                gamma = torch.where(take_sa, torch.full_like(gamma, GAMMA0), gamma)
                stall = torch.where(take_sa, torch.full_like(stall, -ESCAPE_GRACE), stall)
                for si, (seg_lo, seg_hi) in enumerate(bounds):
                    n_sa[si] += int(take_sa[seg_lo:seg_hi].sum())

        if outer % (FREEZE_EVERY_MULT * escape_every) == 0:
            draw = float(rng.random())
            is_boundary = draw < BOUNDARY_FRAC
            if is_boundary:
                fmask = _boundary_mask_t(uv, th, Wu, Hv, a, b, BOUNDARY_DEPTH)
            elif draw < BOUNDARY_FRAC + 0.5 * (1.0 - BOUNDARY_FRAC):
                fmask = _rattler_mask_t(uv, th, Wu, Hv, a, b, pi_full, pj_full,
                                        coarse, refine, RATTLER_K)
            else:
                gsize = int(rng.integers(FREEZE_GROUP_RANGE[0], FREEZE_GROUP_RANGE[1] + 1))
                fmask = _group_mask_t(uv, gsize)
            deep_uv, deep_th = uv, th
            deep_Wu, deep_Hv = Wu, Hv
            deep_ok = torch.zeros(B, dtype=torch.bool, device=dev)
            for _ in range(FREEZE_ROUNDS):
                sig = float(rng.uniform(*SA_KICK_RANGE)) * 0.5
                nudge = torch.randn_like(deep_uv) * sig * fmask.unsqueeze(-1)
                ang_mult = (sig * 4.0) * (BOUNDARY_LR_TH_MULT if is_boundary else 1.0)
                nth = torch.randn_like(deep_th) * ang_mult * fmask
                du, dt, dWu, dHv, dok, _ = _repair_t(
                    deep_uv + nudge, deep_th + nth, deep_Wu, deep_Hv, steps=repair_steps,
                    free_mask=fmask, **(rep_kw_bnd if is_boundary else rep_kw))
                better = dok & (dWu * dHv < deep_Wu * deep_Hv)
                deep_uv = torch.where(dok.view(B, 1, 1), du, deep_uv)
                deep_th = torch.where(dok.view(B, 1), dt, deep_th)
                deep_Wu = torch.where(better, dWu, deep_Wu)
                deep_Hv = torch.where(better, dHv, deep_Hv)
                deep_ok = deep_ok | dok
            if bool(deep_ok.any()):
                uv = torch.where(deep_ok.view(B, 1, 1), deep_uv, uv)
                th = torch.where(deep_ok.view(B, 1), deep_th, th)
                Wu = torch.where(deep_ok, deep_Wu, Wu)
                Hv = torch.where(deep_ok, deep_Hv, Hv)
                for si, (seg_lo, seg_hi) in enumerate(bounds):
                    hits = int(deep_ok[seg_lo:seg_hi].sum())
                    n_deep[si] += hits
                    if is_boundary:
                        n_bnd[si] += hits

        if outer % (JAM_EVERY_MULT * escape_every) == 0:
            jmask, tmask, tcentre = _jam_mask_t(uv, th, Wu, Hv, a, b, JAM_FREE_K)
            jam_uv, jam_th = uv, th
            jam_Wu, jam_Hv = Wu, Hv
            start_area = Wu * Hv
            jam_area = start_area
            jam_ok = torch.zeros(B, dtype=torch.bool, device=dev)
            for _ in range(JAM_ROUNDS):
                if float(rng.random()) < JAM_FLIP_PROB:
                    dth = tmask * (math.pi / 2.0)
                else:
                    dth = torch.randn_like(jam_th) * (math.pi / 4.0) * jmask
                pu = jam_uv + torch.randn_like(jam_uv) * JAM_POS_SIGMA * jmask.unsqueeze(-1)
                pt = jam_th + dth
                ju, jt, jWu, jHv, jok, _ = _repair_t(
                    pu, pt, jam_Wu, jam_Hv, steps=repair_steps, free_mask=jmask, **rep_kw_bnd)
                keep = jok & (jWu * jHv <= jam_area * JAM_SLACK)
                jam_uv = torch.where(keep.view(B, 1, 1), ju, jam_uv)
                jam_th = torch.where(keep.view(B, 1), jt, jam_th)
                jam_Wu, jam_Hv = torch.where(keep, jWu, jam_Wu), torch.where(keep, jHv, jam_Hv)
                jam_area = torch.where(keep, jWu * jHv, jam_area)
                jam_ok = jam_ok | keep
            if bool(jam_ok.any()):
                wide = _group_mask_t(jam_uv, FREEZE_GROUP_RANGE[1], centres=tcentre)
                wide = torch.maximum(wide, jmask)
                wu2, wt2, wWu, wHv, wok, _ = _repair_t(
                    jam_uv, jam_th, jam_Wu, jam_Hv, steps=repair_steps, free_mask=wide, **rep_kw)
                relaxed = wok & (wWu * wHv <= start_area + 1e-9)
                jam_uv = torch.where(relaxed.view(B, 1, 1), wu2, jam_uv)
                jam_th = torch.where(relaxed.view(B, 1), wt2, jam_th)
                jam_Wu = torch.where(relaxed, wWu, jam_Wu)
                jam_Hv = torch.where(relaxed, wHv, jam_Hv)
                take_jam = jam_ok & relaxed
                uv = torch.where(take_jam.view(B, 1, 1), jam_uv, uv)
                th = torch.where(take_jam.view(B, 1), jam_th, th)
                Wu = torch.where(take_jam, jam_Wu, Wu)
                Hv = torch.where(take_jam, jam_Hv, Hv)
                for si, (seg_lo, seg_hi) in enumerate(bounds):
                    n_jam[si] += int(take_jam[seg_lo:seg_hi].sum())

        if outer % (SYM_EVERY_MULT * escape_every) == 0:
            sym_sel = torch.zeros(B, dtype=torch.bool, device=dev)
            for si, (seg_lo, seg_hi) in enumerate(bounds):
                r_lo = reserve_lo[si]
                if r_lo >= seg_hi:
                    if not force_branches:
                        continue
                    r_lo = seg_hi - 1
                sym_hi = r_lo + max(1, (seg_hi - r_lo) // 2)
                sym_sel[r_lo:sym_hi] = True
            if bool(sym_sel.any()):
                su, st_ = _symmetrise_t(uv, th, partner_t)
                su = torch.where(sym_sel.view(B, 1, 1), su, uv)
                st_ = torch.where(sym_sel.view(B, 1), st_, th)
                su2, st2, sWu, sHv, sok, _ = _repair_t(su, st_, Wu, Hv, steps=repair_steps,
                                                       **rep_kw)
                take_sym = sym_sel & sok
                if bool(take_sym.any()):
                    uv = torch.where(take_sym.view(B, 1, 1), su2, uv)
                    th = torch.where(take_sym.view(B, 1), st2, th)
                    Wu, Hv = torch.where(take_sym, sWu, Wu), torch.where(take_sym, sHv, Hv)
                    for si, (seg_lo, seg_hi) in enumerate(bounds):
                        n_sym[si] += int(take_sym[seg_lo:seg_hi].sum())

        if outer % reseed_every == 0:
            dead = stall >= stall_reset_b
            champs = []
            for si, (seg_lo, seg_hi) in enumerate(bounds):
                res_lo = reserve_lo[si]
                for grp_lo, grp_hi in ((seg_lo, res_lo), (res_lo, seg_hi)):
                    if grp_hi - grp_lo < 2:
                        continue
                    g_order = torch.argsort(best_area[grp_lo:grp_hi])
                    g_a = best_area[grp_lo:grp_hi][g_order]
                    g_r = (best_Wu / best_Hv)[grp_lo:grp_hi][g_order]
                    g_clone = ((g_a[1:] - g_a[:-1]).abs() <= g_a[:-1].abs() * DEDUP_REL) & \
                              ((g_r[1:] - g_r[:-1]).abs() <= g_r[:-1].abs() * DEDUP_REL)
                    dead[grp_lo + g_order[1:]] |= g_clone
                main_hi = res_lo if res_lo > seg_lo else seg_hi
                order = torch.argsort(best_area[seg_lo:main_hi])
                champ = seg_lo + int(order[0])
                main_R = main_hi - seg_lo
                for pos in order[-max(1, main_R // 8):]:
                    dst = seg_lo + int(pos)
                    if dst == champ or bool(dead[dst]):
                        continue
                    uv[dst] = best_uv[champ] * KICK + torch.randn_like(uv[dst]) * 0.04
                    th[dst] = best_th[champ] + torch.randn_like(th[dst]) * 0.15
                    Wu[dst], Hv[dst] = best_Wu[champ] * KICK, best_Hv[champ] * KICK
                    best_uv[dst], best_th[dst] = best_uv[champ], best_th[champ]
                    best_Wu[dst], best_Hv[dst] = best_Wu[champ], best_Hv[champ]
                    best_area[dst] = best_area[champ]
                    stall[dst] = 0
                    gamma[dst] = GAMMA0
                dead[champ] = False
                champs.append(champ)
            if force_branches:
                dead = torch.ones_like(dead)
                for c in champs:
                    dead[c] = False

            for si, (seg_lo, seg_hi) in enumerate(bounds):
                idx_all = torch.nonzero(dead[seg_lo:seg_hi], as_tuple=False).flatten() + seg_lo
                if not int(idx_all.numel()):
                    continue

                res_lo = reserve_lo[si]
                idx_res = idx_all[idx_all >= res_lo]
                idx_main = idx_all[idx_all < res_lo]
                order = torch.argsort(best_area[seg_lo:res_lo]) if res_lo > seg_lo \
                    else torch.argsort(best_area[seg_lo:seg_hi])
                elite = seg_lo + order[:min(CROSSOVER_ELITE_K, max(res_lo, seg_lo + 1) - seg_lo)]
                if int(elite.numel()) >= 2 and int(idx_main.numel()):
                    n_cross_slots = int(idx_main.numel() * CROSSOVER_FRAC)
                    perm = idx_main[torch.randperm(int(idx_main.numel()), device=dev)]
                    idx_cross, idx = perm[:n_cross_slots], perm[n_cross_slots:]
                    elite_uv = best_uv.index_select(0, elite).clone()
                    elite_th = best_th.index_select(0, elite).clone()
                    elite_Wu = best_Wu.index_select(0, elite).clone()
                    elite_Hv = best_Hv.index_select(0, elite).clone()
                    n_cross[si] += int(idx_cross.numel())
                    for j in range(int(idx_cross.numel())):
                        dst = int(idx_cross[j])
                        pa, pb = torch.randperm(int(elite.numel()), device=dev)[:2].tolist()
                        cu, ct, cW, cH = _crossover_t(elite_uv, elite_th, elite_Wu, elite_Hv,
                                                      pa, pb, rng)
                        uv[dst], th[dst] = cu, ct
                        Wu[dst], Hv[dst] = cW, cH
                        best_uv[dst], best_th[dst] = cu, ct
                        best_Wu[dst], best_Hv[dst] = cW, cH
                        best_area[dst] = float("inf")
                        gamma[dst] = GAMMA0
                        stall[dst] = 0
                        pw[dst] = 1.0
                else:
                    idx = idx_main

                champ_aspect = float(best_Wu[champs[si]] / best_Hv[champs[si]])
                seeded = []
                for idx_grp, fan in ((idx, _adaptive_fan(champ_aspect, rng)),
                                     (idx_res, ASPECT_FAN)):
                    if not int(idx_grp.numel()):
                        continue
                    m = int(idx_grp.numel())
                    n_reset[si] += m
                    fu, ft, fW, fH = make_population(m + 1, rng, fan=fan)
                    uv.index_copy_(0, idx_grp, torch.tensor(fu[1:], dtype=dtype, device=dev))
                    th.index_copy_(0, idx_grp, torch.tensor(ft[1:], dtype=dtype, device=dev))
                    Wu.index_copy_(0, idx_grp, torch.tensor(fW[1:], dtype=dtype, device=dev))
                    Hv.index_copy_(0, idx_grp, torch.tensor(fH[1:], dtype=dtype, device=dev))
                    seeded.append(idx_grp)
                if not seeded:
                    continue
                idx = torch.cat(seeded) if len(seeded) > 1 else seeded[0]
                best_uv.index_copy_(0, idx, uv.index_select(0, idx))
                best_th.index_copy_(0, idx, th.index_select(0, idx))
                best_Wu.index_copy_(0, idx, Wu.index_select(0, idx))
                best_Hv.index_copy_(0, idx, Hv.index_select(0, idx))
                best_area.index_fill_(0, idx, float("inf"))
                gamma.index_fill_(0, idx, GAMMA0)
                stall.index_fill_(0, idx, 0)
                pw.index_fill_(0, idx, 1.0)

        if migrate is not None and migrate_every and outer % migrate_every == 0:
            world = migrate(extract())
            for si, key in enumerate(keys):
                seg_lo, seg_hi = bounds[si]
                foreign = world.get(key)
                seg = best_area[seg_lo:seg_hi]
                if foreign is None or foreign["area"] >= float(seg.min()) * (1.0 - 1e-9):
                    continue
                pool_hi = reserve_lo[si] if reserve_lo[si] > seg_lo else seg_hi
                dst = seg_lo + int(torch.argmax(best_area[seg_lo:pool_hi]))
                uv[dst] = torch.tensor(foreign["uv"], dtype=dtype, device=dev)
                th[dst] = torch.tensor(foreign["theta"], dtype=dtype, device=dev)
                Wu[dst], Hv[dst] = foreign["Wu"], foreign["Hv"]
                best_uv[dst], best_th[dst] = uv[dst], th[dst]
                best_Wu[dst], best_Hv[dst] = foreign["Wu"], foreign["Hv"]
                best_area[dst] = foreign["Wu"] * foreign["Hv"]
                stall[dst] = 0
                gamma[dst] = GAMMA0
                n_immigrants[si] += 1

        now = time.time()
        if on_checkpoint is not None and now >= next_ck:
            on_checkpoint(extract())
            next_ck = now + checkpoint_every
        if verbose and now >= next_log:
            per = " ".join(f"{k}:{v['area']:.4f}/{implied(v['area']):.1f}"
                           for k, v in extract().items())
            acc_s = [float(ok[seg_lo:seg_hi].float().mean()) for seg_lo, seg_hi in bounds]
            gamma_s = [float(gamma[seg_lo:seg_hi].median()) for seg_lo, seg_hi in bounds]
            stall_s = [int(stall[seg_lo:seg_hi].float().median()) for seg_lo, seg_hi in bounds]
            res_asp = []
            for si, (seg_lo, seg_hi) in enumerate(bounds):
                r_lo = reserve_lo[si]
                if r_lo >= seg_hi:
                    res_asp.append(float("nan"))
                    continue
                r_area = best_area[r_lo:seg_hi]
                fin = torch.isfinite(r_area)
                if not bool(fin.any()):
                    res_asp.append(float("nan"))
                    continue
                r_best = r_lo + int(torch.argmin(torch.where(fin, r_area,
                                                             torch.full_like(r_area, float("inf")))))
                res_asp.append(float(best_Wu[r_best] / best_Hv[r_best]))
            main_asp = []
            for si, (seg_lo, seg_hi) in enumerate(bounds):
                m_hi = reserve_lo[si] if reserve_lo[si] > seg_lo else seg_hi
                m_best = seg_lo + int(torch.argmin(best_area[seg_lo:m_hi]))
                main_asp.append(float(best_Wu[m_best] / best_Hv[m_best]))
            print(f"{pre}[{now - t0:7.1f}s] outer={outer:6d} acc={_csv(acc_s, '.2f')} "
                  f"gamma_med={_csv(gamma_s, '.2e')} stall_med={_csv(stall_s)} "
                  f"esc={_csv(n_escape)} sa={_csv(n_sa)} dp={_csv(n_deep)} bnd={_csv(n_bnd)} jam={_csv(n_jam)} sym={_csv(n_sym)} rst={_csv(n_reset)} "
                  f"imm={_csv(n_immigrants)} cross={_csv(n_cross)} "
                  f"asp={_csv(main_asp, '.3f')} rasp={_csv(res_asp, '.3f')} "
                  f"miss={_csv(miss_ema, '.3f')} {per}")
            next_log = now + log_every

    if verbose:
        print(f"{pre}finished: {outer} outer iterations (per shape, order {keys}): "
              f"esc={_csv(n_escape)} sa={_csv(n_sa)} dp={_csv(n_deep)} bnd={_csv(n_bnd)} jam={_csv(n_jam)} sym={_csv(n_sym)} rst={_csv(n_reset)} "
              f"imm={_csv(n_immigrants)} cross={_csv(n_cross)} "
              f"miss={_csv(miss_ema, '.3f')}, in {time.time() - t0:.1f}s on {dev}")
    return extract()


def run_islands(keys, seconds, replicas_per_shape, devices, seed=0, migrate_every=10,
                on_checkpoint=None, checkpoint_every=180.0, verbose=True, **kw):
    lock = threading.Lock()
    world = {}

    def migrate(local):
        with lock:
            for key, val in local.items():
                if key not in world or val["area"] < world[key]["area"]:
                    world[key] = val
            return {key: dict(val) for key, val in world.items()}

    out = [None] * len(devices)
    failures = [None] * len(devices)

    def island(idx, dev):
        try:
            out[idx] = search(keys, seconds, replicas_per_shape, device=dev,
                              seed=seed + 7919 * idx, migrate=migrate,
                              migrate_every=migrate_every, on_checkpoint=None,
                              tag=str(dev), verbose=verbose, **kw)
        except BaseException as exc:
            failures[idx] = exc
            traceback.print_exc()
            print(f"  [{dev}] ISLAND DIED: {type(exc).__name__}: {exc}")

    threads = [threading.Thread(target=island, args=(i, d), daemon=True)
               for i, d in enumerate(devices)]
    if verbose:
        print(f"  island model: {len(devices)} populations of {replicas_per_shape} replicas/shape "
              f"on {[str(d) for d in devices]}, champion exchange every {migrate_every} outers")
    for t in threads:
        t.start()
    next_ck = time.time() + checkpoint_every
    while any(t.is_alive() for t in threads):
        time.sleep(1.0)
        if time.time() >= next_ck:
            snap = migrate({})
            if snap and on_checkpoint is not None:
                on_checkpoint(snap)
            next_ck = time.time() + checkpoint_every
    for t in threads:
        t.join()
    for res in out:
        if res:
            migrate(res)
    alive = [i for i, r in enumerate(out) if r is not None]
    if not alive:
        raise RuntimeError(f"every island failed; first error: {failures[0]!r}")
    if len(alive) < len(devices):
        print(f"  WARNING: only {len(alive)}/{len(devices)} islands finished; "
              f"the run used {len(alive)} of the available GPUs")
    return migrate({})
