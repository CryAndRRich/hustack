import hashlib
import io
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

DATA = Path(__file__).resolve().parent.parent / "data"
IMG = 64


def read(path):
    raw = path.read_bytes()
    with Image.open(io.BytesIO(raw)) as im:
        assert im.size == (IMG, IMG) and im.mode == "L", f"{path}: {im.size} {im.mode}"
        return np.asarray(im, dtype=np.uint8), hashlib.md5(raw).hexdigest()


def main():
    train = pd.read_csv(DATA / "train.csv")
    test_ids = np.arange(7000)
    assert sorted(int(p.stem) for p in (DATA / "test").glob("*.jpg")) == test_ids.tolist()
    with ThreadPoolExecutor(8) as ex:
        tr = list(ex.map(read, [DATA / "train" / f"{i}.jpg" for i in train["id"]]))
        te = list(ex.map(read, [DATA / "test" / f"{i}.jpg" for i in test_ids]))
    npz = DATA / "fer_cache.npz"
    np.savez_compressed(npz, train_ids=train["id"].to_numpy(), train_labels=train["label"].to_numpy(),
                        X_tr=np.stack([a for a, _ in tr]), md5_tr=np.array([h for _, h in tr]),
                        test_ids=test_ids, X_te=np.stack([a for a, _ in te]), md5_te=np.array([h for _, h in te]))
    z = DATA / "data.zip"
    z.unlink(missing_ok=True)
    subprocess.run(["zip", "-j", "-q", str(z), str(npz)], check=True)
    npz.unlink()
    print(f"{z}: {z.stat().st_size / 1e6:.1f} MB — train {len(tr)}, test {len(te)}")


if __name__ == "__main__":
    main()
