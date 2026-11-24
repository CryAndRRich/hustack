import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.fft import irfft, next_fast_len, rfft, rfftfreq

IG_BAND = (2e-3, 2e-2)


def lstsq(y, xs):
    a = np.stack(xs, axis=1)
    g = np.linalg.lstsq(a, y, rcond=None)[0]
    return g, y - a @ g


def box(v, width):
    if width <= 1:
        return v
    c = np.concatenate([[0], np.cumsum(v)])
    i = np.arange(len(v))
    lo, hi = np.clip(i - width // 2, 0, len(v)), np.clip(i + width // 2 + 1, 0, len(v))
    return (c[hi] - c[lo]) / (hi - lo)


def segment_ffts(x, seg):
    frames = sliding_window_view(x, seg)[::seg // 2]
    return rfft(frames * np.hanning(seg), axis=1)


def transfer(r, p, seg, smooth, phase, weighted, shrink):
    R, P = segment_ffts(r, seg), segment_ffts(p, seg)
    f = rfftfreq(seg)
    band = (f >= IG_BAND[0]) & (f <= IG_BAND[1])
    w = np.ones(len(R))
    for _ in range(2 if weighted else 1):
        sxy = (w[:, None] * R * np.conj(P)).sum(0)
        sxx = (w[:, None] * np.abs(P) ** 2).sum(0)
        sxy = box(sxy.real, smooth) + 1j * box(sxy.imag, smooth)
        sxx = np.maximum(box(sxx, smooth), 1e-30)
        h = 1j * sxy.imag / sxx if phase == "imag" else sxy / sxx
        if weighted:
            w = 1.0 / np.maximum(np.mean(np.abs(R - h * P)[:, band] ** 2, axis=1), 1e-30)
    if shrink:
        syy = np.maximum(box((w[:, None] * np.abs(R) ** 2).sum(0), smooth), 1e-30)
        coh = np.abs(sxy) ** 2 / (sxx * syy)
        keff = w.sum() ** 2 / (w ** 2).sum() * max(smooth, 1)
        h = h * np.clip(1 - 1 / (keff * np.maximum(coh, 1e-12)), 0, 1)
    h[0] = 0
    return f, h


def apply_transfer(p, f, h):
    n = len(p)
    nfft = next_fast_len(2 * n)
    fg = rfftfreq(nfft)
    hg = np.interp(fg, f, h.real) + 1j * np.interp(fg, f, h.imag)
    return irfft(hg * rfft(p, nfft), nfft)[:n]
