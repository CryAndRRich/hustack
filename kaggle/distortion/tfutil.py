import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter


def stft(x, nps):
    _, _, X = signal.stft(x, fs=1.0, window="hann", nperseg=nps, noverlap=3 * nps // 4)
    return X


def istft(X, nps, n):
    _, y = signal.istft(X, fs=1.0, window="hann", nperseg=nps, noverlap=3 * nps // 4)
    return y[:n]


def padded(x, nps, pad):
    return np.pad(x, nps, mode="reflect") if pad else x


def cropped(y, nps, n, pad):
    return y[nps:nps + n] if pad else y[:n]


def through_stft(x, nps, fn, pad):
    xp = padded(x, nps, pad)
    return cropped(istft(fn(stft(xp, nps)), nps, len(xp)), nps, len(x), pad)


def local(p, frames, bins=3):
    return uniform_filter(p, size=(bins, frames), mode="nearest")


def median_db(X):
    return 10 * np.log10(np.maximum(np.median(np.abs(X) ** 2, axis=1), 1e-30))
