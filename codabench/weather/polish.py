import numpy as np
import torch
from numba import cuda

from dp import _dev
from geo import latlon, leg_geometry
from rise import power_vec

INFEASIBLE = 1.0e9
STEPS = (1.0, 0.5, 0.25, 0.1)


def track_terms(win, frame, land, a_deg, b_deg, t_h, wps, cfg, constrained, smooth):
    corr = win.corr
    p = frame.xyz(torch.deg2rad(a_deg), torch.deg2rad(b_deg))
    lat, lon = latlon(p)
    lonn = corr.norm_lon(lon)
    ft, fy, fx = corr.frac(lat, lonn, t_h, clip=False)
    w = win.cubic(torch.clamp(ft, 0, corr.T - 1), torch.clamp(fy, 0, corr.nlat - 1),
                  torch.clamp(fx, 0, corr.nlon - 1))
    dist, sh, ch = leg_geometry(p)
    dt_h = t_h[:, 1:] - t_h[:, :-1]
    vs = dist / (dt_h * 3600.0)
    wp = wps[:, None]

    def pw(sl):
        return power_vec(w[:, sl, 0], w[:, sl, 1], w[:, sl, 2], w[:, sl, 3], w[:, sl, 4], sh, ch, vs, wp, smooth)

    (s0, x0), (s1, x1) = pw(slice(None, -1)), pw(slice(1, None))
    half = 0.5 * dt_h / 1000.0
    e_s = ((s0 + s1) * half).double().sum(1)
    e_x = ((x0 + x1) * half).double().sum(1)
    ok = torch.ones_like(e_x, dtype=torch.bool)
    if not constrained:
        return e_s, e_x, torch.zeros_like(e_s), ok
    relu = torch.relu
    tws = torch.sqrt(w[..., 0] ** 2 + w[..., 1] ** 2 + 1e-12)
    hs = w[..., 2]
    km = land.km(lat, lonn)
    m = cfg["box_margin"]
    pen = (relu(tws - cfg["tws_lim"]) ** 2 + relu(hs - cfg["hs_lim"]) ** 2 + relu(cfg["land_km"] - km) ** 2
           + relu(m - fy) ** 2 + relu(fy - (corr.nlat - 1 - m)) ** 2
           + relu(m - fx) ** 2 + relu(fx - (corr.nlon - 1 - m)) ** 2).double().sum(1)
    f = cfg["feas"]
    ok = ((tws.amax(1) <= f["tws"]) & (hs.amax(1) <= f["hs"]) & (km[:, 1:-1].amin(1) >= f["land_km"])
          & (fy.amin(1) >= 0) & (fy.amax(1) <= corr.nlat - 1) & (fx.amin(1) >= 0) & (fx.amax(1) <= corr.nlon - 1))
    return e_s, e_x, pen, ok


def hessian_blocks(merit_fn, x, free_beta):
    xv = x.detach().requires_grad_(True)
    merit, *extra = merit_fn(xv)
    g, = torch.autograd.grad(merit.sum(), xv, create_graph=True)
    B, N, _ = x.shape
    D = torch.zeros((B, N, 2, 2), dtype=torch.float64, device=x.device)
    L = torch.zeros_like(D)
    col = torch.arange(N, device=x.device) % 3
    prev = (col + 2) % 3
    comps = (0, 1) if free_beta else (0,)
    for r in range(3):
        for c in comps:
            probe = torch.zeros_like(xv)
            probe[:, col == r, c] = 1.0
            hv, = torch.autograd.grad(g, xv, grad_outputs=probe, retain_graph=True)
            hv = hv.detach().double()
            D[:, col == r, :, c] = hv[:, col == r]
            L[:, prev == r, :, c] = hv[:, prev == r]
    L[:, 0] = 0.0
    g = g.detach().double()
    if not free_beta:
        D[..., 0, 1] = D[..., 1, 0] = 0.0
        D[..., 1, 1] = 1.0
        L[..., 1, :] = L[..., :, 1] = 0.0
        g[..., 1] = 0.0
    return merit.detach(), [e.detach() for e in extra], g, D, L


@cuda.jit
def _block_solve_kernel(D, L, g, lam, dx, sinv, y, ok):
    b = cuda.grid(1)
    if b >= D.shape[0]:
        return
    n = D.shape[1]
    lm = lam[b]
    for m in range(n):
        a11 = D[b, m, 0, 0] + lm * (abs(D[b, m, 0, 0]) + 1e-9)
        a22 = D[b, m, 1, 1] + lm * (abs(D[b, m, 1, 1]) + 1e-9)
        a12 = 0.5 * (D[b, m, 0, 1] + D[b, m, 1, 0])
        r1 = -g[b, m, 0]
        r2 = -g[b, m, 1]
        if m > 0:
            l11, l12, l21, l22 = L[b, m, 0, 0], L[b, m, 0, 1], L[b, m, 1, 0], L[b, m, 1, 1]
            s11, s12, s22 = sinv[b, m - 1, 0], sinv[b, m - 1, 1], sinv[b, m - 1, 2]
            t11 = l11 * s11 + l12 * s12
            t12 = l11 * s12 + l12 * s22
            t21 = l21 * s11 + l22 * s12
            t22 = l21 * s12 + l22 * s22
            a11 -= t11 * l11 + t12 * l12
            a12 -= t11 * l21 + t12 * l22
            a22 -= t21 * l21 + t22 * l22
            r1 -= t11 * y[b, m - 1, 0] + t12 * y[b, m - 1, 1]
            r2 -= t21 * y[b, m - 1, 0] + t22 * y[b, m - 1, 1]
        det = a11 * a22 - a12 * a12
        if not (a11 > 0.0 and det > 1e-30 * a11 * a22):
            ok[b] = 0
            return
        sinv[b, m, 0] = a22 / det
        sinv[b, m, 1] = -a12 / det
        sinv[b, m, 2] = a11 / det
        y[b, m, 0] = r1
        y[b, m, 1] = r2
    for m in range(n - 1, -1, -1):
        z1 = y[b, m, 0]
        z2 = y[b, m, 1]
        if m < n - 1:
            d1 = dx[b, m + 1, 0]
            d2 = dx[b, m + 1, 1]
            z1 -= L[b, m + 1, 0, 0] * d1 + L[b, m + 1, 1, 0] * d2
            z2 -= L[b, m + 1, 0, 1] * d1 + L[b, m + 1, 1, 1] * d2
        dx[b, m, 0] = sinv[b, m, 0] * z1 + sinv[b, m, 1] * z2
        dx[b, m, 1] = sinv[b, m, 1] * z1 + sinv[b, m, 2] * z2
    ok[b] = 1


def block_solve(D, L, g, lam):
    B, N = g.shape[:2]
    dx = torch.zeros_like(g)
    sinv = torch.empty((B, N, 3), dtype=torch.float64, device=g.device)
    y = torch.empty((B, N, 2), dtype=torch.float64, device=g.device)
    ok = torch.zeros(B, dtype=torch.uint8, device=g.device)
    D, L, g, lam = D.contiguous(), L.contiguous(), g.contiguous(), lam.contiguous()
    torch.cuda.synchronize()
    _block_solve_kernel[(B + 63) // 64, 64](_dev(D), _dev(L), _dev(g), _dev(lam), _dev(dx), _dev(sinv), _dev(y),
                                            _dev(ok))
    cuda.synchronize()
    return dx, ok.bool()


def newton(win, frame, land, alpha, beta, t_h, wps, free_beta, constrained, cfg, stages, trace=None):
    dev = win.coef.device
    f32 = dict(dtype=torch.float32, device=dev)
    a0 = torch.tensor(np.degrees(alpha), **f32)
    b0 = torch.tensor(np.degrees(beta), **f32)
    th = torch.tensor(t_h, **f32)
    wp = torch.tensor(np.asarray(wps, dtype=np.float32), **f32)
    B = len(a0)
    x = torch.stack([a0[:, 1:-1], b0[:, 1:-1]], -1)

    def terms(xx, smooth):
        aa = torch.cat([a0[:, :1], xx[..., 0], a0[:, -1:]], 1)
        bb = torch.cat([b0[:, :1], xx[..., 1], b0[:, -1:]], 1)
        return track_terms(win, frame, land, aa, bb, th, wp, cfg, constrained, smooth)

    best = torch.full((B,), float("inf"), dtype=torch.float64, device=dev)
    best_x = x.clone()

    def keep(xx, e_x, ok):
        score = torch.where(ok, e_x, e_x + INFEASIBLE)
        better = score < best
        best.copy_(torch.where(better, score, best))
        best_x[better] = xx[better]

    lam = torch.full((B,), cfg["lam0"], dtype=torch.float64, device=dev)
    it = 0
    for tau, dz, w_pen, iters in stages:
        def merit_fn(xx):
            e_s, e_x, pen, ok = terms(xx, (tau, dz))
            return e_s + w_pen * pen, e_x, ok

        for _ in range(iters):
            merit, (e_x, ok), g, D, L = hessian_blocks(merit_fn, x, free_beta)
            keep(x, e_x, ok)
            dx, pd = block_solve(torch.nan_to_num(D), torch.nan_to_num(L), torch.nan_to_num(g), lam)
            dx = torch.nan_to_num(dx, nan=0.0, posinf=0.0, neginf=0.0)
            dx = dx * (cfg["max_step_deg"] / dx.abs().amax((1, 2)).clamp_min(1e-30)).clamp_max(1.0)[:, None, None]
            step = torch.zeros_like(lam)
            m_best = merit.clone()
            with torch.no_grad():
                for t in STEPS:
                    xt = x + (t * dx).float()
                    m_t, e_t, ok_t = merit_fn(xt)
                    keep(xt, e_t, ok_t)
                    better = pd & (m_t < m_best)
                    m_best = torch.where(better, m_t, m_best)
                    step = torch.where(better, torch.full_like(step, t), step)
            x = x + (step[:, None, None] * dx).float()
            lam = torch.where(step == 1.0, lam / 3.0, torch.where(step > 0, lam, lam * 10.0)).clamp(1e-8, 1e8)
            it += 1
        if trace is not None:
            feas = best < INFEASIBLE
            n0, e0 = trace.get(it, (0, 0.0))
            trace[it] = (n0 + int(feas.sum()), e0 + float(best[feas].sum()))
    with torch.no_grad():
        _, e_x, _, ok = terms(x, stages[-1][:2])
        keep(x, e_x, ok)
    a_out = torch.cat([a0[:, :1], best_x[..., 0], a0[:, -1:]], 1).double().cpu().numpy()
    b_out = torch.cat([b0[:, :1], best_x[..., 1], b0[:, -1:]], 1).double().cpu().numpy()
    return np.radians(a_out), np.radians(b_out)


def newton_selftest(win, frame, land, cfg):
    dev = win.coef.device
    M, n = 14, 12
    t_h = torch.tensor(float(win.a + 40) + np.arange(M + 1) * 0.25, dtype=torch.float32, device=dev)[None]
    a_full = torch.linspace(0.0, 0.6, M + 1, device=dev)[None]
    b_full = (0.3 * torch.sin(torch.linspace(0.0, 3.0, M + 1, device=dev)))[None]
    a_full[:, -1] = 0.62
    wp = torch.ones(1, device=dev)

    def merit_fn(xx):
        aa = torch.cat([a_full[:, :1], xx[..., 0], a_full[:, -1:]], 1)
        bb = torch.cat([b_full[:, :1], xx[..., 1], b_full[:, -1:]], 1)
        e_s, e_x, pen, ok = track_terms(win, frame, land, aa, bb, t_h, wp, cfg, False, (5.0, 0.02))
        return e_s, e_x, ok

    x0 = torch.stack([a_full[:, 1:-1], b_full[:, 1:-1]], -1)
    _, _, g, D, L = hessian_blocks(merit_fn, x0, True)
    dense = torch.autograd.functional.hessian(lambda v: merit_fn(v.reshape(1, M - 1, 2))[0].sum(),
                                              x0.reshape(-1)).double()
    blocks = torch.zeros_like(dense)
    for m in range(M - 1):
        blocks[2 * m:2 * m + 2, 2 * m:2 * m + 2] = D[0, m]
        if m:
            blocks[2 * m:2 * m + 2, 2 * m - 2:2 * m] = L[0, m]
            blocks[2 * m - 2:2 * m, 2 * m:2 * m + 2] = L[0, m].T
    err_h = float((blocks - dense).abs().max() / dense.abs().max().clamp_min(1e-12))
    assert err_h < 1e-3, f"Hessian blocks mismatch {err_h}"
    rng = np.random.default_rng(0)
    Bt = 3
    Dn = rng.normal(size=(Bt, n, 2, 2))
    Dn = Dn @ np.swapaxes(Dn, -1, -2) + 6.0 * np.eye(2)
    Ln = rng.normal(size=(Bt, n, 2, 2))
    Ln[:, 0] = 0.0
    gn = rng.normal(size=(Bt, n, 2))
    lam_n = np.array([0.0, 0.1, 1.0])
    t64 = lambda a: torch.tensor(a, dtype=torch.float64, device=dev)
    dx, pd = block_solve(t64(Dn), t64(Ln), t64(gn), t64(lam_n))
    err_s = 0.0
    for k in range(Bt):
        A = np.zeros((2 * n, 2 * n))
        for m in range(n):
            A[2 * m:2 * m + 2, 2 * m:2 * m + 2] = Dn[k, m] + lam_n[k] * np.diag(np.abs(np.diag(Dn[k, m])) + 1e-9)
            if m:
                A[2 * m:2 * m + 2, 2 * m - 2:2 * m] = Ln[k, m]
                A[2 * m - 2:2 * m, 2 * m:2 * m + 2] = Ln[k, m].T
        ref = np.linalg.solve(A, -gn[k].reshape(-1))
        assert bool(pd[k]) == bool(np.all(np.linalg.eigvalsh(A) > 0)), "positive-definiteness flag wrong"
        if pd[k]:
            err_s = max(err_s, float(np.abs(dx[k].cpu().numpy().reshape(-1) - ref).max() / np.abs(ref).max()))
    assert err_s < 1e-8, f"block solver mismatch {err_s}"
    return err_h, err_s
