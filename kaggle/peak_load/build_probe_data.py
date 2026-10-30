import zipfile

import numpy as np
import pandas as pd

from build_probes import EXTRA

BEST = EXTRA / "v1.4" / "blend" / "submission.csv"
ZIP = EXTRA.parent / "data" / "peak-load-probe.zip"


def main():
    v11 = pd.read_csv(EXTRA / "v1.1" / "submission.csv")
    best = pd.read_csv(BEST)
    assert (v11.id == best.id).all()
    corr = best.load_mw.values - v11.load_mw.values
    assert np.all(corr[v11.load_mw.values >= 1200] == 0)
    df = pd.DataFrame({"id": v11.id, "v11": v11.load_mw, "corr": corr})
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("probe_correction.csv", df.to_csv(index=False))
    print(f"wrote {ZIP}: {len(df)} rows, rms corr {np.sqrt(np.mean(corr**2)):.3f}")


if __name__ == "__main__":
    main()
