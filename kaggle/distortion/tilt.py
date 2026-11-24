import itertools

import numpy as np

from tfutil import cropped, istft, local, padded, stft

TIDE_PERIODS_H = [12.421, 23.934, 12.0, 25.819, 12.658, 24.066, 11.967, 13.661 * 12]
BLOCK_S = 1800
WF_NPS = [1024, 4096]
WF_BETA = [1.0, 4.0, 16.0]
WF_FRAMES = 3


def block_angles(h1, h2, block=BLOCK_S):
    n = len(h1)
    edges = np.append(np.arange(0, n, block), n)
    if edges[-1] - edges[-2] < block // 2 and len(edges) > 2:
        edges = np.delete(edges, -2)
    mids, angs = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        x, y = h1[a:b], h2[a:b]
        angs.append(0.5 * np.arctan2(2 * x @ y, x @ x - y @ y))
        mids.append((a + b) / 2)
    angs = np.unwrap(2 * np.array(angs)) / 2
    return np.interp(np.arange(n), mids, angs)


def tide_angle(h1, h2, iters=30):
    t = np.arange(len(h1), dtype=np.float64)
    w = 2 * np.pi / (np.array(TIDE_PERIODS_H) * 3600)
    x = np.column_stack([np.ones_like(t)] + [fn(wk * t) for wk in w for fn in (np.sin, np.cos)])
    theta = np.linalg.solve(x.T @ x + 1e-9 * len(t) * np.eye(x.shape[1]), x.T @ block_angles(h1, h2))
    ang = x @ theta
    start = np.sum((h2 * np.cos(ang) - h1 * np.sin(ang)) ** 2)
    for _ in range(iters):
        e = h2 * np.cos(ang) - h1 * np.sin(ang)
        jac = -(h2 * np.sin(ang) + h1 * np.cos(ang))[:, None] * x
        jtj = jac.T @ jac
        step = np.linalg.solve(jtj + 1e-9 * np.trace(jtj) / len(jtj) * np.eye(len(jtj)), -jac.T @ e)
        theta += step
        ang = x @ theta
        if np.abs(step).max() < 1e-12:
            break
    return ang, float(np.sum((h2 * np.cos(ang) - h1 * np.sin(ang)) ** 2) / start)


def detilt_modes(z, h1, h2, a, b, ang, igz):
    cs, sn = np.cos(ang), np.sin(ang)
    a_hat, e = h1 * cs + h2 * sn, h2 * cs - h1 * sn
    k = a * cs + b * sn
    yield "raw", z - a * h1 - b * h2 - igz
    yield "proj", z - k * a_hat - igz
    for nps, beta in itertools.product(WF_NPS, WF_BETA):
        yield "wf_n%d_b%g" % (nps, beta), z - k * wiener_tilt(a_hat, e, nps, beta) - igz


def wiener_tilt(a_hat, e, nps, beta):
    n = len(a_hat)
    A, E = stft(padded(a_hat, nps, True), nps), stft(padded(e, nps, True), nps)
    w = np.clip(1 - beta * local(np.abs(E) ** 2, WF_FRAMES) / np.maximum(local(np.abs(A) ** 2, WF_FRAMES), 1e-30),
                0.0, 1.0)
    return cropped(istft(w * A, nps, n + 2 * nps), nps, n, True)
