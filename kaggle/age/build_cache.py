import zipfile

import numpy as np

from preprocess import CACHE, IMAGE_DIRS, decode, load_tables


def main():
    d, lab, judge, _ = load_tables()
    out = d / "age-cache.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for part, df in (("train", lab), ("judge", judge)):
            images = decode([d / IMAGE_DIRS[part] / p for p in df.path])
            images_file, ids_file = CACHE[part]
            for name, arr in ((images_file, images), (ids_file, df.ID.to_numpy())):
                with z.open(name, "w", force_zip64=True) as f:
                    np.save(f, arr)
            print(f"{part}: {images.shape} {images.dtype}")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
