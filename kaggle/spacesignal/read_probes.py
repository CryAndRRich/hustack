import pandas as pd

from build_probes import COLS, K, ROOT

N = 100
CELLS = 3 * N
LB_SAMPLE = 1.36441
LB_V1 = 1.02730


def main():
    sc = pd.read_csv(ROOT / "data" / "probes" / "scores.csv").dropna()
    s = {r.probe: float(r.lb) * CELLS for r in sc.itertuples()}
    v1 = pd.read_csv(ROOT / "extra" / "v1" / "submission.csv")
    sample = pd.read_csv(ROOT / "data" / "raw" / "sample_sub.csv")
    rows = {}
    pc = [(c, s.get(f"p{1 + 2 * i:02d}_{c}_plusK"), s.get(f"p{2 + 2 * i:02d}_{c}_minusK")) for i, c in enumerate(COLS)]
    if all(p is not None and m is not None for _, p, m in pc):
        rest = {c: (p + m - 2 * N * K) / 2 for c, p, m in pc}
        A = sum(rest.values()) / 2
        for c, p, m in pc:
            rows[c] = {"mean_T": (m - p) / 2 / N, "mean_absT": (A - rest[c]) / N}
        print(f"sum |T| from the +-K probes: {A:.4f} (mean {A / CELLS:.5f})")
        if "p00_zeros" in s:
            print(f"  check vs zeros probe: {s['p00_zeros']:.4f} (diff {s['p00_zeros'] - A:+.4f}; > 0.01 means a"
                  " |T| >= K or not plain MAE)")
    elif "p00_zeros" in s:
        A = s["p00_zeros"]
        print(f"sum |T| from the zeros probe: {A:.4f} (mean {A / CELLS:.5f})")
    else:
        A = None
    for i, c in enumerate(COLS):
        r = rows.setdefault(c, {})
        r["V1_mean"], r["V1_mean_abs"] = float(v1[c].mean()), float(v1[c].abs().mean())
        if A is not None and "mean_absT" in r:
            a_c = r["mean_absT"] * N
            for tag, key in (("v1", f"p{7 + 2 * i:02d}_{c}_v1"), ("v1neg", f"p{8 + 2 * i:02d}_{c}_v1neg")):
                if key in s:
                    r[f"mae_{tag}"] = (s[key] - (A - a_c)) / N
            r["mae_zero"] = r["mean_absT"]
    print(pd.DataFrame(rows).T.round(5).to_string())
    print("per column: mae_v1 << mae_zero means V1's column carries the key's row values; mae_v1neg << mae_v1 means the"
          " key uses the opposite sign; mean_T / mean_absT vs V1_mean / V1_mean_abs shows scale and sign of the key.")
    if A is not None and all("mae_v1" in rows[c] for c in COLS):
        tot = sum(rows[c]["mae_v1"] for c in COLS) / 3
        print(f"V1 file predicted from the column probes: {tot:.5f} (LB {LB_V1}); must agree to ~1e-5")
    r2 = {k: v / N for k, v in s.items() if k.startswith("r2_")}
    if r2:
        print("round 2 (alpha_V1, 2 beta_V1, gamma candidate): gamma MAE assuming alpha and beta exact"
              " (r2_beta2 must read 2.82788 / 3 = 0.94263 for that)")
        for k, v in r2.items():
            print(f"  {k:22s} LB {v / 3:.5f} -> gamma MAE {v:.5f}")
    print(f"known readings: sample_sub {LB_SAMPLE} (values {sample[COLS].to_numpy().min():.3f}..{sample[COLS].to_numpy().max():.3f}),"
          f" V1 {LB_V1}")


if __name__ == "__main__":
    main()
