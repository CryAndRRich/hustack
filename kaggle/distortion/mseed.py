import datetime as dt
from pathlib import Path

import numpy as np

STEIM1 = 10


def _record_start(h):
    year, doy = int.from_bytes(h[20:22], "big"), int.from_bytes(h[22:24], "big")
    t = dt.datetime(year, 1, 1, h[24], h[25], h[26], tzinfo=dt.timezone.utc) + dt.timedelta(days=doy - 1)
    return t.timestamp() + int.from_bytes(h[28:30], "big") * 1e-4


def _record_layout(raw):
    h = raw[:64]
    blk = int.from_bytes(h[46:48], "big")
    while blk:
        kind, nxt = int.from_bytes(raw[blk:blk + 2], "big"), int.from_bytes(raw[blk + 2:blk + 4], "big")
        if kind == 1000:
            return raw[blk + 4], raw[blk + 5], 2 ** raw[blk + 6]
        blk = nxt
    raise ValueError("no blockette 1000")


def _steim1_diffs(words):
    words = words.astype(np.uint32)
    nib = (words[..., :1] >> (30 - 2 * np.arange(16, dtype=np.uint32))) & 3
    nib[:, 0, 1:3] = 0
    nib[..., 0] = 0
    b8 = ((words[..., None] >> np.array([24, 16, 8, 0], np.uint32)) & 0xFF).astype(np.uint8).view(np.int8)
    b16 = ((words[..., None] >> np.array([16, 0], np.uint32)) & 0xFFFF).astype(np.uint16).view(np.int16)
    cand = np.zeros(words.shape + (4,), np.int64)
    valid = np.zeros(words.shape + (4,), bool)
    m1, m2, m3 = nib == 1, nib == 2, nib == 3
    cand[m1] = b8[m1]
    valid[m1] = True
    cand[m2, :2] = b16[m2]
    valid[m2, :2] = True
    cand[m3, 0] = words.view(np.int32)[m3]
    valid[m3, 0] = True
    nrec = words.shape[0]
    return cand.reshape(nrec, -1), valid.reshape(nrec, -1)


def read_mseed(path):
    raw = Path(path).read_bytes()
    enc, order, reclen = _record_layout(raw)
    assert enc == STEIM1 and order == 1, (path, enc, order)
    assert len(raw) % reclen == 0, path
    recs = np.frombuffer(raw, np.uint8).reshape(-1, reclen)
    offsets = {int.from_bytes(bytes(r[44:46]), "big") for r in recs}
    assert len(offsets) == 1, offsets
    off = offsets.pop()
    words = recs[:, off:].copy().view(">u4").reshape(len(recs), -1, 16)
    x0 = words[:, 0, 1].astype(np.uint32).view(np.int32).astype(np.int64)
    xn = words[:, 0, 2].astype(np.uint32).view(np.int32).astype(np.int64)
    cand, valid = _steim1_diffs(words)
    counts = valid.sum(axis=1)
    npts = np.array([int.from_bytes(bytes(r[30:32]), "big") for r in recs])
    assert (counts >= npts).all(), path
    flat = cand[valid]
    seg_start = np.concatenate([[0], np.cumsum(counts)[:-1]])
    out_start = np.concatenate([[0], np.cumsum(npts)[:-1]])
    take = np.repeat(seg_start - out_start, npts) + np.arange(npts.sum())
    seq = flat[take]
    seq[out_start] = x0
    csum = np.cumsum(seq)
    base = np.repeat(np.concatenate([[0], csum[out_start[1:] - 1]]), npts)
    samples = csum - base
    assert (samples[out_start + npts - 1] == xn).all(), path

    chans = [bytes(r[15:18]).decode().strip() for r in recs]
    starts = np.array([_record_start(bytes(r[:30])) for r in recs])
    rates = {(int.from_bytes(bytes(r[32:34]), "big", signed=True), int.from_bytes(bytes(r[34:36]), "big", signed=True))
             for r in recs}
    assert rates == {(1, 1)}, rates
    out = {}
    for ch in sorted(set(chans)):
        idx = sorted((i for i, c in enumerate(chans) if c == ch), key=lambda i: starts[i])
        for a, b in zip(idx[:-1], idx[1:]):
            assert abs(starts[b] - starts[a] - npts[a]) < 1e-3, (path, ch, a, b)
        out[ch] = np.concatenate([samples[out_start[i]:out_start[i] + npts[i]] for i in idx])
    return out
