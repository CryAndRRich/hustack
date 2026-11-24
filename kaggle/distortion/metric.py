import numpy as np

NOISE_FLOOR = 64.0
WINDOW_STARTS = [8640 * i for i in range(10)]
WINDOW = 600
SPLITS = {"public": [0, 2, 4, 6, 8], "private": [1, 3, 5, 7, 9], "all": list(range(10))}


def window_index(split="all"):
    return np.concatenate([np.arange(WINDOW_STARTS[w], WINDOW_STARTS[w] + WINDOW) for w in SPLITS[split]])


def _detrended_rms(diff, index):
    slope, offset = np.polyfit(index, diff, 1)
    return float(np.sqrt(np.mean((diff - offset - slope * index) ** 2)))


def column_score(sub, ref, noisy, index):
    err_sub = _detrended_rms(sub[index] - ref[index], index)
    err_noisy = _detrended_rms(noisy[index] - ref[index], index)
    best = 20 * np.log10(err_noisy / NOISE_FLOOR)
    achieved = 20 * np.log10(err_noisy / err_sub) if err_sub > 0 else np.inf
    return float(min(100.0, 100.0 * achieved / best))


def scores(sub, ref, noisy):
    out = {s: column_score(sub, ref, noisy, window_index(s)) for s in SPLITS}
    out["full"] = column_score(sub, ref, noisy, np.arange(len(ref)))
    return out
