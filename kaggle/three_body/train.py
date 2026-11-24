import json
import shutil
import signal
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from preprocess import (TARGETS, TEST_LIKE_RMIN0, energy_angmom, find_data, initial, rmin0, rmse, row_rms, run_ref,
                        stack, subset)
from integrators import ARITH_VARIANTS, CANDIDATES, predict

WORK = Path("/tmp/work")
OUT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
N0 = 7000
NEWTON_ITERS = 4
SCAN_TOP = 3
SCAN_HALF = 40
FULL_TOP = 5
EXACT_SMOOTH = 1e-3
CE_RMIN = 0.05
EXACT_TL_RMSE = 0.01
RUN_BUDGET_S = 900
TOTAL_BUDGET_S = 4 * 3600
SMOKE = {"rows": (6, 3), "n0": 200, "iters": 1, "half": 2, "top": 2}
HOST_CALIB = 2.966e-4


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()

    def flush(self):
        for st in self.streams:
            st.flush()


class Timeout(Exception):
    pass


def on_alarm(signum, frame):
    raise Timeout()


class Runs:
    def __init__(self, deadline):
        self.deadline, self.failed = deadline, []

    def __call__(self, label, fn, quiet=False):
        if time.time() > self.deadline:
            print(f"{label}: skipped, past the global deadline")
            return None
        t0 = time.time()
        signal.alarm(RUN_BUDGET_S)
        try:
            out = fn()
        except Timeout:
            print(f"{label}: skipped, over its {RUN_BUDGET_S} s budget")
            return None
        except Exception as e:
            print(f"{label}: failed {e!r}")
            self.failed.append(label)
            return None
        finally:
            signal.alarm(0)
        if not quiet:
            print(f"{label}: {time.time() - t0:.1f} s", flush=True)
        return out


def match(pred, truth, ref):
    e = row_rms(pred, truth)
    return float(np.median(e / np.maximum(row_rms(ref, truth), 1e-300))), int((e < 1e-12).sum())


def calibrate(st_s, ref_s, truth_s, cfg, runs):
    print(f"== calibrate N on {len(truth_s)} smooth rows")
    e_host = truth_s - ref_s
    rows = []
    for c in CANDIDATES:
        n, path = cfg["n0"], []
        for _ in range(cfg["iters"]):
            out = runs(f"calib {c} N={n}", lambda: predict(c, st_s, n), quiet=True)
            if out is None:
                break
            e_c = out[0] - ref_s
            k = (e_host * e_c).sum(1) / (e_c ** 2).sum(1)
            n_r = n * np.abs(k) ** -0.25
            path.append(n)
            n = int(round(float(np.median(n_r))))
        out = runs(f"calib {c} N={n}", lambda: predict(c, st_s, n), quiet=True)
        if out is None:
            continue
        e_c = out[0] - ref_s
        k = (e_host * e_c).sum(1) / (e_c ** 2).sum(1)
        med, n_exact = match(out[0], truth_s, ref_s)
        rows.append({"cand": c, "N": n, "path": "→".join(map(str, path + [n])), "k_pos_share": float(np.mean(k > 0)),
                     "mad_logN": float(np.median(np.abs(np.log(np.abs(k)) / 4 - np.median(np.log(np.abs(k)) / 4)))),
                     "match": med, "rows_exact": n_exact})
    A = pd.DataFrame(rows, columns=["cand", "N", "path", "k_pos_share", "mad_logN", "match", "rows_exact"])
    A = A.sort_values("match").reset_index(drop=True)
    print(A.to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    return A


def scan(A, st_s, ref_s, truth_s, cfg, runs):
    print("== integer scan of N")
    rows = []
    for _, r in A.head(cfg["top"]).iterrows():
        for n in range(max(1, int(r.N) - cfg["half"]), int(r.N) + cfg["half"] + 1):
            out = runs(f"scan {r.cand} N={n}", lambda: predict(r.cand, st_s, n), quiet=True)
            if out is not None:
                med, n_exact = match(out[0], truth_s, ref_s)
                rows.append({"cand": r.cand, "N": n, "match": med, "rows_exact": n_exact})
    S = pd.DataFrame(rows, columns=["cand", "N", "match", "rows_exact"]).sort_values("match").reset_index(drop=True)
    for c, g in S.groupby("cand"):
        b = g.iloc[0]
        print(f"{c}: best N={b.N} match {b.match:.3e} exact rows {b.rows_exact}/{len(truth_s)}")
    return S


def evaluate_full(cands_n, state, truth, ntr, test_like, ref, ref2, i_calib, runs):
    print("== full runs")
    m0, p0, v0, _ = subset(state, slice(0, ntr))
    E0, L0 = energy_angmom(m0, p0, v0)
    variants = []
    for c, n in cands_n:
        out = runs(f"full {c} N={n}", lambda: predict(c, state, n))
        if out is None or not np.isfinite(out[0]).all():
            continue
        pred, rmin = out
        e = row_rms(pred[:ntr], truth)
        E1, L1 = energy_angmom(m0, pred[:ntr, :9].reshape(-1, 3, 3), pred[:ntr, 9:].reshape(-1, 3, 3))
        v = {"name": f"{c}|N={n}", "cand": c, "N": n, "pred": pred, "rmin": rmin,
             "all": rmse(pred[:ntr], truth), "test_like": rmse(pred[:ntr][test_like], truth[test_like]),
             "rows_lt_1e-9": int((e < 1e-9).sum()), "rows_lt_1e-6": int((e < 1e-6).sum()),
             "dL_max": float(np.abs(L1 - L0).max()), "dE_rel_med": float(np.median(np.abs(E1 - E0) / np.abs(E0))),
             "calib": float(np.linalg.norm(pred[i_calib, :9] - ref2[i_calib, :9]))}
        variants.append(v)
        print(f"   all {v['all']:.4e} test-like {v['test_like']:.4e} <1e-9 {v['rows_lt_1e-9']} <1e-6 {v['rows_lt_1e-6']} "
              f"dL_max {v['dL_max']:.1e} dE_med {v['dE_rel_med']:.1e} calib {v['calib']:.4e} (host {HOST_CALIB:.4e})",
              flush=True)
    variants.sort(key=lambda x: x["all"])
    if variants:
        b = variants[0]
        err, err_ref = row_rms(b["pred"][:ntr], truth), row_rms(ref[:ntr], truth)
        worst = np.argsort(-err_ref)[:12]
        print(f"rows with the largest ref error, against {b['name']}:")
        print(pd.DataFrame({"row": worst, "err_ref": err_ref[worst], "err_best": err[worst],
                            "rmin_traj": b["rmin"][worst]}).to_string(index=False, float_format=lambda x: f"{x:.3e}"))
    return variants


def hedge(variants, ref, state, smooth_exact, runs):
    best = variants[0]
    if smooth_exact and best["cand"] in ARITH_VARIANTS["fr_DKD"]:
        members = [best["pred"]]
        for c in ARITH_VARIANTS["fr_DKD"]:
            if c != best["cand"]:
                out = runs(f"hedge {c} N={best['N']}", lambda: predict(c, state, best["N"]))
                if out is not None and np.isfinite(out[0]).all():
                    members.append(out[0])
    else:
        members = [v["pred"] for v in variants[:FULL_TOP]] + [ref]
    ce = best["rmin"] < CE_RMIN
    out = best["pred"].copy()
    out[ce] = np.mean(members, axis=0)[ce]
    print(f"hedge: {len(members)} members on {ce.sum()} close-encounter rows (rmin < {CE_RMIN})")
    return out, ce


def choose(arms, truth, ntr, test_like):
    score = {k: {"all": rmse(a[:ntr], truth), "test_like": rmse(a[:ntr][test_like], truth[test_like])}
             for k, a in arms.items()}
    print("== arms (train RMSE)")
    for k, s in score.items():
        print(f"{k:6s} all {s['all']:.4e} test-like {s['test_like']:.4e}")
    ref = score["ref"]
    if "best" in score and score["best"]["test_like"] < EXACT_TL_RMSE:
        pick = "best"
    else:
        wins = [k for k in ("best", "hedge") if k in score and score[k]["test_like"] < ref["test_like"]
                and score[k]["all"] < ref["all"]]
        pick = min(wins, key=lambda k: score[k]["test_like"]) if wins else "ref"
    print(f"chosen by the pre-registered rule: {pick}")
    if pick == "ref":
        print("NO NEW SUBMISSION NEEDED: same model as the scored LB 0.17126")
    return pick, score


def write_arm(path, ids, pred):
    d = pd.DataFrame(pred, columns=TARGETS)
    d.insert(0, "Id", ids)
    d.to_csv(path, index=False)
    return d


def pipeline(tr, te, sample, out_dir, cfg, smoke, runs):
    out_dir.mkdir(parents=True, exist_ok=True)
    ntr = len(tr)
    st_tr, st_te = initial(tr), initial(te)
    state = stack(st_tr, st_te)
    truth = tr[TARGETS].to_numpy(float)
    test_like = rmin0(st_tr) > TEST_LIKE_RMIN0
    print(f"rows: train {ntr} (test-like {test_like.sum()}), test {len(te)}")
    ref = runs("ref DOP853 1e-13", lambda: run_ref(state, 1e-13))
    ref2 = runs("ref2 DOP853 1e-12", lambda: run_ref(state, 1e-12))
    assert ref is not None and ref2 is not None, "reference integration failed"
    e_host, noise = row_rms(truth, ref[:ntr]), row_rms(ref, ref2)[:ntr]
    smooth = np.ones(ntr, bool) if smoke else (e_host > 1e-11) & (e_host < 1e-6) & (noise < 0.01 * e_host)
    i_calib = ntr + int(np.argmax(st_te[3]))
    st_s, ref_s, truth_s = subset(st_tr, smooth), ref[:ntr][smooth], truth[smooth]

    A = calibrate(st_s, ref_s, truth_s, cfg, runs)
    S = scan(A, st_s, ref_s, truth_s, cfg, runs)
    top = S.drop_duplicates("cand").head(FULL_TOP)
    variants = evaluate_full(list(zip(top.cand, top.N)), state, truth, ntr, test_like, ref, ref2, i_calib, runs)

    arms = {"ref": ref}
    if variants:
        arms["best"] = variants[0]["pred"]
        smooth_exact = bool(S.iloc[0].match < EXACT_SMOOTH)
        print(f"smooth rows reproduced (match < {EXACT_SMOOTH}): {smooth_exact}")
        arms["hedge"], ce = hedge(variants, ref, state, smooth_exact, runs)
    pick, score = choose(arms, truth, ntr, test_like)

    extra = out_dir / "extra"
    extra.mkdir(exist_ok=True)
    A.to_csv(out_dir / "calibrate.csv", index=False)
    S.to_csv(out_dir / "scan.csv", index=False)
    pd.DataFrame([{k: v for k, v in x.items() if k not in ("pred", "rmin")} for x in variants]).to_csv(
        out_dir / "full.csv", index=False)
    for k, a in arms.items():
        write_arm(extra / f"train_{k}.csv", tr.Id.to_numpy(), a[:ntr])
        write_arm(extra / f"test_{k}.csv", te.Id.to_numpy(), a[ntr:])
    if variants:
        pd.DataFrame({"Id": np.concatenate([tr.Id, te.Id]), **{f"rmin_{x['name']}": x["rmin"] for x in variants},
                      "close_encounter": ce}).to_csv(extra / "rmin.csv", index=False)

    sub = write_arm(out_dir / "submission.csv", te.Id.to_numpy(), arms[pick][ntr:])
    assert list(sub.columns) == list(sample.columns), "column order differs from sample_submission.csv"
    assert (sub.Id.to_numpy() == sample.Id.to_numpy()).all(), "Id order differs from sample_submission.csv"
    assert np.isfinite(sub[TARGETS].to_numpy()).all(), "non-finite prediction"
    return {"pick": pick, "score": score, "calibrate": A.to_dict("records"),
            "full": [{k: v for k, v in x.items() if k not in ("pred", "rmin")} for x in variants]}


def main():
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    log = open(WORK / "log.txt", "w")
    sys.stdout = Tee(sys.__stdout__, log)
    signal.signal(signal.SIGALRM, on_alarm)
    summary, ok = {}, False
    try:
        data = find_data()
        tr, te = pd.read_csv(data / "train.csv"), pd.read_csv(data / "test.csv")
        sample = pd.read_csv(data / "sample_submission.csv")
        print(f"data {data}: train {tr.shape}, test {te.shape}")
        assert len(tr) == 250 and len(te) == 100, "unexpected row counts"
        runs = Runs(time.time() + TOTAL_BUDGET_S)

        print("== smoke")
        a, b = SMOKE["rows"]
        pipeline(tr.head(a), te.head(b), sample.head(b), WORK / "smoke", SMOKE, True, runs)
        assert not runs.failed, f"smoke: failed runs {runs.failed}"
        shutil.rmtree(WORK / "smoke")
        print("smoke ok")

        print("== full run")
        cfg = {"n0": N0, "iters": NEWTON_ITERS, "half": SCAN_HALF, "top": SCAN_TOP}
        summary = pipeline(tr, te, sample, WORK, cfg, False, runs)
        shutil.copy(WORK / "submission.csv", OUT / "submission.csv")
        ok = True
    finally:
        (WORK / "results.json").write_text(json.dumps(summary, indent=1, default=float))
        sys.stdout = sys.__stdout__
        log.close()
        with zipfile.ZipFile(OUT / "results.zip", "w", zipfile.ZIP_DEFLATED) as z:
            for f in WORK.rglob("*"):
                if f.is_file() and f.name != "submission.csv":
                    z.write(f, f.relative_to(WORK))
    if ok:
        print(f"SUBMIT: {OUT / 'submission.csv'}")


if __name__ == "__main__":
    main()
