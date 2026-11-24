import numpy as np

from signals import apply_transfer, lstsq, transfer
from tfutil import median_db, stft, through_stft
from tilt import tide_angle

COMPL_CFG = dict(seg=4096, smooth=7, phase="free", weighted=True, shrink=True)
V1_NPS, V1_BETA = 4096, 4.0
REF_BAND = (0.9e-3, 1.6e-3)


def prepare(ch):
    z, h1, h2, p = ch["LHZ"], ch["LH1"], ch["LH2"], ch["LDG"]
    gains, r = lstsq(z, [h1, h2])
    ang, tide_gain = tide_angle(h1, h2)
    return {"a": float(gains[0]), "b": float(gains[1]), "igz": apply_transfer(p, *transfer(r, p, **COMPL_CFG)),
            "ang": ang, "tide_gain": tide_gain}


def true_nos(ch, land, compl):
    n1, n2 = ch["LH1"] - land["LH1"], ch["LH2"] - land["LH2"]
    (a, b), res = lstsq(ch["LHZ"] - land["LHZ"] - compl, [n1, n2])
    return res / (1 - a - b)


def ref_band():
    f = np.fft.rfftfreq(V1_NPS)
    return (f >= REF_BAND[0]) & (f <= REF_BAND[1])


def template_db(x):
    db = median_db(stft(x, V1_NPS))
    return db - db[ref_band()].mean()


def v1_wiener(u, template, pad=False):
    band = ref_band()

    def gain(X):
        noise = 10 ** ((template + median_db(X)[band].mean()) / 10)
        return np.maximum(0.0, 1 - V1_BETA * noise[:, None] / np.maximum(np.abs(X) ** 2, 1e-30)) * X

    return through_stft(u, V1_NPS, gain, pad)
