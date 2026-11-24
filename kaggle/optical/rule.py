import numpy as np
from sklearn.neighbors import NearestNeighbors

LOCKED = ["beam_b", "mirror_a", "pressure_b", "gate_a"]
LO = np.array([110.0, -1.2, 40000.0, 0.002])
HI = np.array([200.0, 1.2, 48000.0, 0.038])
K = 9
N_CLASSES = 3


def scaled(df):
    return (df[LOCKED].to_numpy(float) - LO) / (HI - LO)


def vote(archive, runs):
    scores = np.zeros((len(runs), N_CLASSES))
    for brand in runs["cartridge"].unique():
        a = (archive["cartridge"] == brand).to_numpy()
        r = (runs["cartridge"] == brand).to_numpy()
        assert a.sum() >= K, brand
        d, i = NearestNeighbors(n_neighbors=K, metric="chebyshev").fit(scaled(archive)[a]).kneighbors(scaled(runs)[r])
        y = archive["profile"].to_numpy()[a][i]
        w = 1.0 / np.maximum(d, 1e-12)
        scores[r] = np.stack([(w * (y == c)).sum(1) for c in range(N_CLASSES)], 1)
    share = scores / scores.sum(1, keepdims=True)
    return share.argmax(1), share.max(1)
