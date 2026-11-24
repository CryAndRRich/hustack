import hashlib
import http.client
import json
import shutil
import time
import urllib.request
import zipfile
from pathlib import Path

RECORD = "https://zenodo.org/api/records/23470870"
DATASET = "weather-era5"
NC_FILES = [f"era5_{kind}_{basin}_{period}.nc"
            for basin in ("atlantic", "pacific") for period in ("2024", "2025_01") for kind in ("wind", "waves")]
LAND_FILES = [f"ne_10m_land.{ext}" for ext in ("shp", "shx", "dbf", "prj")]
CHUNK = 16 << 20
RETRIES = 30
OUT = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path(".")


def get(url, start=0):
    headers = {"Range": f"bytes={start}-"} if start else {}
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120)


def record_files():
    with get(RECORD) as r:
        meta = json.load(r)
    return {f["key"]: (f["size"], f["links"]["self"]) for f in meta["files"]}


def parse_sums(text):
    sums = {}
    for line in text.splitlines():
        if line.strip():
            digest, name = line.split()
            sums[name.lstrip("*")] = digest.lower()
    return sums


def stream(zf, name, size, url, digest):
    sha, done, failures, t0 = hashlib.sha256(), 0, 0, time.time()
    next_report = 1 << 30
    with zf.open(name, "w", force_zip64=True) as dst:
        while done < size:
            try:
                with get(url, done) as r:
                    if done and r.status != 206:
                        raise RuntimeError(f"{name}: server ignored the Range request (status {r.status})")
                    while chunk := r.read(CHUNK):
                        dst.write(chunk)
                        sha.update(chunk)
                        done += len(chunk)
                        if done >= next_report:
                            print(f"  {name}: {done / 1e9:.2f}/{size / 1e9:.2f} GB, "
                                  f"{done / 1e6 / (time.time() - t0):.0f} MB/s", flush=True)
                            next_report += 1 << 30
            except (OSError, http.client.HTTPException) as e:
                failures += 1
                if failures > RETRIES:
                    raise
                print(f"  {name}: {type(e).__name__} at {done} bytes, retry {failures}/{RETRIES}", flush=True)
                time.sleep(min(60, 5 * failures))
    assert done == size, f"{name}: got {done} bytes, expected {size}"
    assert sha.hexdigest() == digest, f"{name}: SHA-256 mismatch"
    print(f"{name}: {size / 1e9:.3f} GB, SHA-256 ok, {time.time() - t0:.0f} s", flush=True)


def main():
    files = record_files()
    with get(files["SHA256SUMS"][1]) as r:
        sums_text = r.read().decode()
    sums = parse_sums(sums_text)
    wanted = NC_FILES + LAND_FILES
    missing = [n for n in wanted if n not in files or n not in sums]
    assert not missing, f"not in the record or SHA256SUMS: {missing}"
    total = sum(files[n][0] for n in wanted)
    print(f"{len(wanted)} files, {total / 1e9:.2f} GB; free disk {shutil.disk_usage(OUT).free / 1e9:.1f} GB", flush=True)

    path = OUT / f"{DATASET}.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as zf:
        zf.writestr("SHA256SUMS", sums_text)
        for name in wanted:
            stream(zf, name, files[name][0], files[name][1], sums[name])

    with zipfile.ZipFile(path) as zf:
        bad = zf.testzip()
        assert bad is None, f"CRC error in {bad}"
        sizes = {i.filename: i.file_size for i in zf.infolist()}
    assert all(sizes[n] == files[n][0] for n in wanted)
    print(f"{path.name}: {path.stat().st_size / 1e9:.2f} GB zipped, {len(sizes)} members, CRC ok")


if __name__ == "__main__":
    main()
