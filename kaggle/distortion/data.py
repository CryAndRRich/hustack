import numpy as np

from mseed import read_mseed

TRAINING = ["MAIN"] + [f"MASS{i:03d}" for i in range(100)]


def _float(chans, n):
    return {k: v[:n].astype(np.float64) for k, v in chans.items()}


def load_training(root, name):
    d = root / "TRAINING" / ("MAIN" if name == "MAIN" else "MASS")
    noisy = read_mseed(d / f"{name}_noisy.mseed")
    land = read_mseed(d / f"{name}_REF_clean_for_eq.mseed")
    compl = read_mseed(d / f"{name}_REF_clean_for_compl.mseed")["LHZ"]
    n = min([len(compl)] + [len(v) for v in noisy.values()] + [len(v) for v in land.values()])
    return {"name": name, "ch": _float(noisy, n), "land": _float(land, n), "compl": compl[:n].astype(np.float64)}


def load_challenge(root, name):
    noisy = read_mseed(root / "CHALLENGE" / name / f"{name}_noisy.mseed")
    return {"name": name, "ch": _float(noisy, min(len(v) for v in noisy.values()))}
