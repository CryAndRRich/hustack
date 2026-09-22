import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from geometry import (HEX67_SCORE, N_ELLIPSES, SHAPE_AB, SHAPES, certify, parse_submission_text,
                      self_test, symmetry_residual, validate, write_submission)
from seeds import hex67
from optimize import DEFAULT_REPLICAS, run_islands, search

AUTO_SMOKE = True
SMOKE_SECONDS = 40.0
TIME_BUDGET_SEC = float(os.environ.get("PACK_BUDGET", 3600))
DEVICE = os.environ.get("PACK_DEVICE") or None
USE_ALL_GPUS = True
REPLICAS_PER_SHAPE = int(os.environ.get("PACK_REPLICAS", 0)) or None
REPAIR_STEPS = 120
CHECKPOINT_EVERY = 180.0
SEED = 0


def working_dir():
    kaggle = Path("/kaggle/working")
    return kaggle if kaggle.exists() else Path(".")


def load_warm():
    warm, warm_score = {}, {}
    for root in (Path("."), Path("/kaggle/input")):
        if not root.exists():
            continue
        for path in sorted(root.rglob("best_packing*.json")):
            try:
                blob = json.loads(path.read_text())
                for key, v in blob["packings"].items():
                    a, b = SHAPE_AB[key]
                    uv, theta = np.array(v["uv"]), np.array(v["theta"])
                    Wu, Hv = float(v["Wu"]), float(v["Hv"])
                    _, _, _, _, rep = certify(a, b, Wu, Hv, uv, theta)
                    if not rep["ok"]:
                        continue
                    if key not in warm_score or rep["score"] > warm_score[key]:
                        warm[key] = {"uv": uv, "theta": theta, "Wu": Wu, "Hv": Hv}
                        warm_score[key] = rep["score"]
                print(f"found checkpoint {path} (recorded score {blob.get('score', float('nan')):.2f})")
            except Exception as exc:
                print(f"ignoring unreadable checkpoint {path}: {exc}")
    if not warm:
        print("no checkpoint found -- cold start from structured seeds")
        return None
    print(f"warm start merged from the checkpoint(s) above: {sorted(warm)}")
    return warm


def hex_floor():
    uv, Wu, Hv = hex67()
    theta = np.zeros(N_ELLIPSES)
    out = {}
    for key, a, b in SHAPES:
        W, H, xy, deg, rep = certify(a, b, Wu, Hv, uv, theta)
        assert rep["ok"], f"hex floor failed to certify for {key}: {rep}"
        out[key] = (W, H, xy, deg, rep)
    return out


def upgrade_floor(floor, warm):
    if not warm:
        return floor
    out = dict(floor)
    for key, a, b in SHAPES:
        w = warm.get(key)
        if w is None:
            continue
        W, H, xy, deg, rep = certify(a, b, w["Wu"], w["Hv"], w["uv"], w["theta"])
        if rep["ok"] and rep["score"] > floor[key][4]["score"]:
            out[key] = (W, H, xy, deg, rep)
            print(f"  warm start raises the {key} floor to {rep['score']:.2f}")
        else:
            why = rep.get("reason", "does not beat the hex floor") if not rep["ok"] else \
                f"scores {rep['score']:.2f}, below the hex floor"
            print(f"  warm start for {key} ignored: {why}")
    return out


def finalise(results, floor):
    packings, rows = {}, []
    for key, a, b in SHAPES:
        W, H, xy, deg, rep = floor[key]
        source = "hex floor"
        cand = results.get(key) if results else None
        if cand is not None:
            cW, cH, cxy, cdeg, crep = certify(a, b, cand["Wu"], cand["Hv"],
                                              cand["uv"], cand["theta"])
            if crep["ok"] and crep["score"] > rep["score"]:
                W, H, xy, deg, rep = cW, cH, cxy, cdeg, crep
                source = "search"
            elif not crep["ok"]:
                print(f"  {key}: search result failed to certify ({crep.get('reason')}), "
                      f"falling back to hex floor")
        packings[key] = (W, H, xy, deg)
        rows.append((key, source, rep))
    return packings, rows


def mean_score(rows):
    return float(np.mean([rep["score"] for _, _, rep in rows]))


def write_and_verify(packings, path):
    write_submission(packings, path)
    parsed = parse_submission_text(pd.read_csv(path)["submission_text"].iloc[0])
    assert set(parsed) == {k for k, _, _ in SHAPES}, f"blocks on disk: {sorted(parsed)}"
    scores = []
    for key, a, b in SHAPES:
        W, H, xy, deg = parsed[key]
        rep = validate(a, b, W, H, xy, deg)
        assert rep["ok"], f"{key} invalid after round trip through disk: {rep}"
        scores.append(rep["score"])
    return float(np.mean(scores)), scores


def save_checkpoint(results, path, score):
    payload = {"schema": 1, "score": score, "saved": time.strftime("%Y-%m-%d %H:%M:%S"),
               "packings": {k: {"uv": v["uv"].tolist(), "theta": v["theta"].tolist(),
                                "Wu": v["Wu"], "Hv": v["Hv"], "area": v["area"]}
                            for k, v in results.items()}}
    Path(path).write_text(json.dumps(payload))


def report(rows, mean_score, label):
    print(f"\n{label}")
    print(f"  {'shape':6} {'source':10} {'W':>14} {'H':>14} {'density':>10} {'score':>12}")
    for key, source, rep in rows:
        print(f"  {key:6} {source:10} {rep['W']:14.8f} {rep['H']:14.8f} "
              f"{rep['density']:10.6f} {rep['score']:12.2f}")
    print(f"  mean score = {mean_score:.2f}   "
          f"(hex floor {HEX67_SCORE:.2f}, LB #1 on 2026-08-22 = 758817.44)")


def main():
    t0 = time.time()
    out = working_dir()
    sub_path, ck_path = out / "submission.csv", out / "best_packing.json"

    print("=" * 78)
    self_test()
    warm = load_warm()
    floor = upgrade_floor(hex_floor(), warm)

    packings, rows = finalise(None, floor)
    score, _ = write_and_verify(packings, sub_path)
    report(rows, score, "floor written to submission.csv")

    print("  point-symmetry residual of the floor being carried into this run (not a target):")
    for key, a, b in SHAPES:
        _, _, xy, _, _ = floor[key]
        sym = symmetry_residual(xy, b)
        print(f"    {key}: median={sym['median']:.4f} ({sym['median_over_b']:.3f} x b) "
              f"max={sym['max']:.4f}")

    keys = [k for k, _, _ in SHAPES]

    import torch
    on_gpu = torch.cuda.is_available() if DEVICE is None else DEVICE.startswith("cuda")
    if on_gpu and not torch.cuda.is_available():
        print("  DEVICE asked for cuda but none is visible -- falling back to cpu")
        on_gpu = False
    replicas = ({k: REPLICAS_PER_SHAPE for k in keys} if REPLICAS_PER_SHAPE
                else DEFAULT_REPLICAS if on_gpu else {k: 12 for k in keys})
    if on_gpu:
        n = torch.cuda.device_count() if USE_ALL_GPUS else 1
        devices = [f"cuda:{i}" for i in range(n)]
    else:
        devices = ["cpu"]
    print(f"\ndevice = {devices} ({torch.cuda.get_device_name(0) if on_gpu else 'cpu'}), "
          f"replicas/shape = {replicas} per island, budget = {TIME_BUDGET_SEC:.0f}s")
    print(f"  per-shape log fields below (acc, gamma_med, stall_med, esc, sa, dp, bnd, jam, sym, rst, imm, cross, "
          f"asp, rasp, miss) are "
          f"comma-separated in this order: {keys}")
    print("  asp/rasp = container aspect (Wu/Hv) of the main population's champion and of the "
          "exploration reserve's own best (V9): if rasp tracks asp all run, the reserve is not "
          "actually exploring a different basin and RESERVE_FRAC/ASPECT_FAN need revisiting")
    print("  dp = local-deepening successes, of which bnd = the boundary-band subset (V11); sym = "
          "symmetry projections kept. bnd/dp should sit near BOUNDARY_FRAC if the band is no harder "
          "to improve than the interior, and clearly below it if it is harder (FINDINGS #31.1)")

    def run(seconds, reps, on_ck, ck_every, **kw):
        if len(devices) > 1:
            return run_islands(keys, seconds, reps, devices, on_checkpoint=on_ck,
                               checkpoint_every=ck_every, **kw)
        return search(keys, seconds, reps, device=devices[0], on_checkpoint=on_ck,
                      checkpoint_every=ck_every, **kw)

    if AUTO_SMOKE:
        print("\n=== AUTO_SMOKE: whole path at trivial settings, result discarded ===")
        smoke_hits = []
        smoke = run(SMOKE_SECONDS, 4,
                    lambda r: smoke_hits.append(
                        write_and_verify(finalise(r, floor)[0], out / "smoke_submission.csv")[0]),
                    5.0, repair_steps=20, coarse=7, refine=2, seed=SEED + 1, warm=warm,
                    force_branches=True, migrate_every=2,
                    escape_every=2, reseed_every=3, log_every=10.0)
        sp, srows = finalise(smoke, floor)
        sm, _ = write_and_verify(sp, out / "smoke_submission.csv")
        assert abs(sm - mean_score(srows)) < 1e-6
        save_checkpoint(smoke, out / "smoke_packing.json", sm)
        report(srows, sm, "smoke result (discarded)")
        print(f"  smoke checkpoints written: {len(smoke_hits)}")
        for p in ("smoke_submission.csv", "smoke_packing.json"):
            (out / p).unlink(missing_ok=True)
        print("=== AUTO_SMOKE passed ===\n")

    best = {"score": score, "packings": packings, "rows": rows, "results": None}

    def consider(results, tag):
        cand_pack, cand_rows = finalise(results, floor)
        m = mean_score(cand_rows)
        if m <= best["score"] + 1e-9:
            print(f"  [{tag}] {m:.2f} does not beat {best['score']:.2f}; submission.csv untouched")
            return False
        disk, _ = write_and_verify(cand_pack, sub_path)
        assert abs(disk - m) < 1e-6, f"disk score {disk} != computed {m}"
        save_checkpoint(results, ck_path, m)
        best.update(score=m, packings=cand_pack, rows=cand_rows, results=results)
        print(f"  [{tag}] submission.csv updated, mean score {m:.2f} "
              f"({m - HEX67_SCORE:+.2f} vs the aligned ceiling)")
        return True

    remaining = TIME_BUDGET_SEC - (time.time() - t0)
    results = run(max(30.0, remaining), replicas,
                  lambda r: consider(r, "checkpoint"), CHECKPOINT_EVERY,
                  repair_steps=REPAIR_STEPS, seed=SEED, warm=warm)
    consider(results, "final")

    disk, _ = write_and_verify(best["packings"], sub_path)
    assert abs(disk - best["score"]) < 1e-6, f"disk {disk} != best {best['score']}"
    report(best["rows"], best["score"], "FINAL (best seen anywhere in this run)")
    print(f"\nwrote {sub_path}" +
          (f" and {ck_path}" if best["results"] is not None
           else " (no checkpoint file: the search never beat the hex floor)") +
          f" in {time.time() - t0:.0f}s total")
    print(f"gap to the aligned ceiling: {best['score'] - HEX67_SCORE:+.2f} points")
    return best["packings"]


def plot(packings):
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse, Rectangle
    except Exception as exc:
        print(f"plot skipped: {exc}")
        return
    fig, axes = plt.subplots(1, len(SHAPES), figsize=(20, 7))
    for ax, (key, a, b) in zip(np.atleast_1d(axes), SHAPES):
        W, H, xy, deg = packings[key]
        ax.add_patch(Rectangle((0, 0), W, H, fill=False, edgecolor="black", lw=1.5))
        for (x, y), d in zip(xy, deg):
            ax.add_patch(Ellipse((x, y), 2 * a, 2 * b, angle=d,
                                 facecolor="steelblue", edgecolor="navy", alpha=0.65, lw=0.4))
        dens = N_ELLIPSES * math.pi * a * b / (W * H)
        ax.set_xlim(-1, W + 1)
        ax.set_ylim(-1, H + 1)
        ax.set_aspect("equal")
        ax.set_title(f"{key}  {W:.3f} x {H:.3f}\ndensity {dens:.6f}  score {(dens*1000)**2:.1f}")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot(main())
