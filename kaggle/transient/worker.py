import json
import pickle
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from config import Config
from models import catboost_fit_predict, protocol, tabpfn_fit_predict
from seqnet import fit_predict_factory


def main(families, data_dir, out_dir):
    start = time.time()
    log = lambda msg: print(f"[{time.time() - start:6.0f}s] {msg}", flush=True)
    data = pickle.load(open(Path(data_dir) / "data.pkl", "rb"))
    cfg = Config(**json.loads((Path(data_dir) / "config.json").read_text()))
    for family in families.split(","):
        try:
            if family == "catboost":
                fp = catboost_fit_predict
            elif family == "tabpfn":
                fp = tabpfn_fit_predict
            elif family == "gru":
                seq = np.load(Path(data_dir) / "seq.npz")
                fp = fit_predict_factory({"train": seq["train"], "test": seq["test"]}, seq["scale"])
            else:
                raise ValueError(family)
            log(f"{family}: start")
            out = protocol(cfg, fp, data, lambda m: log(f"{family}: {m}"))
            for k, v in out.items():
                assert np.isfinite(v).all(), f"{family} {k} has non-finite predictions"
            np.savez(Path(out_dir) / f"{family}.npz", **out)
            log(f"{family}: saved")
        except Exception:
            log(f"{family}: FAILED\n{traceback.format_exc()}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
