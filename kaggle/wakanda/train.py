import json
import shutil
import sys
import time
import traceback
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from preprocess import load, project, residualise
from models import ASH_GRID, THRESH, ash_draw, ols, run_arms

WORK = Path("/tmp/work")
OUT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
ARMS = ["ols", "eb", "ss", "hs", "blend"]
FULL = {"iter": 6000, "burn": 1500, "chains": 2, "sim_iter": 3000, "sim_burn": 800, "sim_chains": 1, "reps": 7}
SMOKE = {"iter": 60, "burn": 20, "chains": 2, "sim_iter": 40, "sim_burn": 10, "sim_chains": 1, "reps": 1}
T0 = time.time()


class Tee:
    def __init__(self, path):
        self.f = open(path, "w")
        self.out = sys.stdout

    def write(self, s):
        self.out.write(s)
        self.f.write(s)

    def flush(self):
        self.out.flush()
        self.f.flush()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def q(x, qs=(0, 0.1, 0.5, 0.9, 1)):
    return " ".join(f"{v:.3g}" for v in np.quantile(x, qs))


def explore(data):
    y, G, C, df = data["y"], data["G"], data["C"], data["df"]
    n, p = G.shape
    sst = float(((y - y.mean()) ** 2).sum())
    S = project(y, data)
    o = ols(S)
    S_lin = project(y, data, data["C_lin"])
    o_lin = ols(S_lin)
    rss, rss_lin = o["sigma2"] * o["dfr"], o_lin["sigma2"] * o_lin["dfr"]
    k_extra = S_lin["dof"] - S["dof"]
    F = ((rss_lin - rss) / k_extra) / o["sigma2"]
    yt_c = residualise(y[:, None], C)[:, 0]
    yt_lin = residualise(y[:, None], data["C_lin"])[:, 0]
    e = {"n": n, "p": p, "y_mean": float(y.mean()), "y_sd": float(y.std()),
         "gene_rate_q": np.quantile(G.mean(0), [0, 0.1, 0.5, 0.9, 1]).tolist(),
         "r2_full": 1 - rss / sst, "r2_demo_only": 1 - float(yt_c @ yt_c) / sst,
         "r2_demo_linear_only": 1 - float(yt_lin @ yt_lin) / sst,
         "sigma_full": float(np.sqrt(o["sigma2"])), "sigma_linear_demo": float(np.sqrt(o_lin["sigma2"])),
         "F_nonlinear_demo": float(F), "F_nonlinear_p": float(stats.f.sf(F, k_extra, o["dfr"]))}
    log(f"y mean {e['y_mean']:.2f} sd {e['y_sd']:.2f}; gene activation rate q0/10/50/90/100: {q(G.mean(0))}")
    for c in df.columns[-4:]:
        log(f"  {c}: {q(df[c].to_numpy(float))}")
    log(f"R2 full {e['r2_full']:.4f}, demographics only {e['r2_demo_only']:.4f} (linear {e['r2_demo_linear_only']:.4f}); "
        f"residual sd {e['sigma_full']:.4f} (linear demo {e['sigma_linear_demo']:.4f}); exact-structure test: "
        f"{'EXACT' if e['sigma_full'] < 0.01 else 'noisy'}")
    log(f"nonlinear demographic terms: F={F:.2f} on {k_extra} df, p={e['F_nonlinear_p']:.3g}")
    t = o["bh"] / o["se"]
    pv = 2 * stats.t.sf(np.abs(t), o["dfr"])
    e.update({"abs_t_q": np.quantile(np.abs(t), [0.5, 0.9, 0.99, 1]).tolist(),
              "n_abs_t_gt": {str(c): int((np.abs(t) > c).sum()) for c in (2, 3, 4, 6)},
              "pi0_storey": float(min(1.0, (pv > 0.5).mean() / 0.5)),
              "n_abs_bh_ge_1": int((np.abs(o["bh"]) >= THRESH).sum()),
              "se_q": np.quantile(o["se"], [0, 0.5, 1]).tolist(),
              "ols_prob_sum": float(o["prob"].sum())})
    log(f"OLS |t| q50/90/99/100 {q(np.abs(t), (0.5, 0.9, 0.99, 1))}; |t|>2/3/4/6: {e['n_abs_t_gt']}; "
        f"Storey pi0 {e['pi0_storey']:.3f}; |bh|>=1: {e['n_abs_bh_ge_1']}; se q0/50/100 {q(o['se'], (0, 0.5, 1))}")
    R = np.corrcoef(G.T)
    off = np.abs(R[~np.eye(p, dtype=bool)])
    ev = np.linalg.eigvalsh(R)[::-1]
    e.update({"gene_corr_abs_mean": float(off.mean()), "gene_corr_abs_max": float(off.max()),
              "gene_pairs_abs_corr_gt_0.3": int((off > 0.3).sum() // 2), "gene_pairs_abs_corr_gt_0.6": int((off > 0.6).sum() // 2),
              "gene_eig_top10_share": (ev[:10] / p).tolist()})
    log(f"gene |corr| mean {off.mean():.3f} max {off.max():.3f}; pairs >0.3: {e['gene_pairs_abs_corr_gt_0.3']}, "
        f">0.6: {e['gene_pairs_abs_corr_gt_0.6']}; top-10 eigen share {' '.join(f'{v:.3f}' for v in ev[:10] / p)}")
    D = df[["sex", "age", "height", "weight"]].to_numpy(float)
    Gc = (G - G.mean(0)) / G.std(0)
    Dc = (D - D.mean(0)) / D.std(0)
    RD = Gc.T @ Dc / n
    Gi = np.column_stack([np.ones(n), G])
    r2_demo_on_genes = []
    for i in range(4):
        coef, *_ = np.linalg.lstsq(Gi, D[:, i], rcond=None)
        res = D[:, i] - Gi @ coef
        r2_demo_on_genes.append(float(1 - res @ res / ((D[:, i] - D[:, i].mean()) ** 2).sum()))
    e.update({"gene_demo_abs_corr_max": np.abs(RD).max(0).tolist(), "gene_demo_n_abs_corr_gt_0.2": (np.abs(RD) > 0.2).sum(0).tolist(),
              "demo_on_genes_r2_in_sample": r2_demo_on_genes})
    log(f"gene-demographic |corr| max per sex/age/height/weight {np.round(np.abs(RD).max(0), 3).tolist()}, "
        f"#genes >0.2 {e['gene_demo_n_abs_corr_gt_0.2']}; demographic ~ 500 genes in-sample R2 {np.round(r2_demo_on_genes, 3).tolist()} "
        f"(noise floor ~{p / n:.2f})")
    Dd = np.corrcoef(D.T)
    log(f"demographic correlations:\n{np.round(Dd, 3)}")
    yc = y - y.mean()
    marg = Gc.T @ yc / n / G.std(0)
    e["marginal_vs_adjusted_corr"] = float(np.corrcoef(marg, o["bh"])[0, 1])
    e["marginal_abs_q"] = np.quantile(np.abs(marg), [0.5, 0.9, 1]).tolist()
    log(f"marginal slope vs adjusted OLS coefficient corr {e['marginal_vs_adjusted_corr']:.3f}; |marginal| q50/90/100 {q(np.abs(marg), (0.5, 0.9, 1))}")
    X = np.column_stack([C, G])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    fit = X @ coef
    res = y - fit
    Z = np.column_stack([np.ones(n), (fit - fit.mean()) ** 2, (fit - fit.mean()) ** 3])
    zc, *_ = np.linalg.lstsq(Z, res, rcond=None)
    rr = res - Z @ zc
    e["resid_on_fit23_r2"] = float(1 - rr @ rr / (res @ res))
    e["resid_skew"], e["resid_kurt"] = float(stats.skew(res)), float(stats.kurtosis(res))
    e["y_decimals_check"] = float(np.abs(y * 100 - np.round(y * 100)).max())
    log(f"residual ~ fit^2, fit^3 R2 {e['resid_on_fit23_r2']:.4f}; residual skew {e['resid_skew']:.3f} excess kurt {e['resid_kurt']:.3f}")
    top = np.argsort(-np.abs(t))[:25]
    log("top-25 |t| genes (gene, bh, se, t, marginal): " + "; ".join(
        f"{j + 1}:{o['bh'][j]:.2f}/{o['se'][j]:.2f}/{t[j]:.1f}/{marg[j]:.2f}" for j in top))
    return e


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def simulate(data, fits, cfg, seed):
    rng = np.random.default_rng(seed)
    G = data["G"]
    n, p = G.shape
    truths = []
    for fam in ("ss", "hs"):
        dr = fits[fam][0]["draws"]
        for i in np.linspace(0, len(dr) - 1, cfg["reps"]).round().astype(int):
            truths.append((fam, dr[i][0], dr[i][1]))
    for _ in range(cfg["reps"]):
        truths.append(("eb", ash_draw(fits["eb"]["pi"], p, rng), fits["eb"]["sigma2"]))
    rows, pooled = [], {a: [] for a in ARMS}
    sim_cfg = {"iter": cfg["sim_iter"], "burn": cfg["sim_burn"], "chains": cfg["sim_chains"]}
    for r, (fam, beta, s2) in enumerate(truths):
        ys = data["C"] @ rng.standard_normal(data["C"].shape[1]) + G @ beta + np.sqrt(s2) * rng.standard_normal(n)
        lab = (np.abs(beta) >= THRESH).astype(float)
        probs, _ = run_arms(project(ys, data), sim_cfg, seed + 1000 * (r + 1))
        row = {"rep": r, "family": fam, "n_pos": int(lab.sum()), "sigma": float(np.sqrt(s2)), "brier_base_rate": brier(lab.mean(), lab)}
        for a in ARMS:
            row[a] = brier(probs[a], lab)
            pooled[a].append(np.column_stack([probs[a], lab]))
        rows.append(row)
        log(f"sim {r:2d} {fam}: pos {row['n_pos']:3d} sigma {row['sigma']:.2f} | " + " ".join(f"{a} {row[a]:.4f}" for a in ARMS))
    return pd.DataFrame(rows), {a: np.vstack(v) for a, v in pooled.items()}


def calibration(pl, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(pl[:, 0], edges) - 1, 0, bins - 1)
    return [{"bin": f"{edges[b]:.1f}-{edges[b + 1]:.1f}", "n": int((idx == b).sum()),
             "mean_p": float(pl[idx == b, 0].mean()) if (idx == b).any() else None,
             "rate": float(pl[idx == b, 1].mean()) if (idx == b).any() else None} for b in range(bins)]


def run(cfg, outdir, seed=2026):
    outdir.mkdir(parents=True, exist_ok=True)
    data = load()
    summary = {"cfg": cfg, "explore": explore(data)}
    log("fitting all arms on the real data")
    probs, fits = run_arms(project(data["y"], data), cfg, seed)
    for fam in ("ss", "hs"):
        if len(fits[fam]) > 1:
            dif = np.abs(fits[fam][0]["prob"] - fits[fam][1]["prob"])
            summary[f"{fam}_chain_prob_absdiff_max"] = float(dif.max())
            log(f"{fam}: chain disagreement max {dif.max():.4f} mean {dif.mean():.4f}")
    ss0, hs0 = fits["ss"][0], fits["hs"][0]
    summary["real_fit"] = {"ss_sigma": float(np.sqrt(ss0["sigma2"])), "ss_w": ss0["w"], "ss_tau": float(np.sqrt(ss0["tau2"])),
                           "ss_k": ss0["k"], "hs_sigma": float(np.sqrt(hs0["sigma2"])), "hs_tau": float(np.sqrt(hs0["tau2"])),
                           "eb_pi": dict(zip(map(str, ASH_GRID.tolist()), fits["eb"]["pi"].round(4).tolist())),
                           "expected_positives": {a: float(probs[a].sum()) for a in ARMS},
                           "self_brier": {a: float((probs[a] * (1 - probs[a])).mean()) for a in ARMS}}
    log(f"real fit: {json.dumps(summary['real_fit'])}")
    log("simulation")
    sim, pooled = simulate(data, fits, cfg, seed + 7)
    sim.to_csv(outdir / "sim.csv", index=False)
    mean = {a: float(sim[a].mean()) for a in ARMS}
    by_fam = sim.groupby("family")[ARMS + ["brier_base_rate"]].mean()
    log(f"simulated Brier by family:\n{by_fam.round(4).to_string()}")
    pick = min(ARMS, key=mean.get)
    summary.update({"sim_brier_mean": mean, "sim_brier_by_family": by_fam.to_dict(), "pick": pick,
                    "calibration": {a: calibration(pooled[a]) for a in ARMS}})
    log(f"simulated Brier mean: {json.dumps({a: round(v, 4) for a, v in mean.items()})} -> pick {pick}")
    for row in summary["calibration"][pick]:
        log(f"  calib {pick} {row}")
    extra = outdir / "extra"
    extra.mkdir(exist_ok=True)
    o = fits["ols"]
    pd.DataFrame({"colID": np.arange(1, len(o["bh"]) + 1), "ols_beta": o["bh"], "ols_se": o["se"],
                  "ss_mean": np.mean([f["post_mean"] for f in fits["ss"]], 0), "ss_pip": np.mean([f["pip"] for f in fits["ss"]], 0),
                  "hs_mean": np.mean([f["post_mean"] for f in fits["hs"]], 0), "eb_mean": fits["eb"]["post_mean"],
                  **{f"p_{a}": probs[a] for a in ARMS}}).to_csv(extra / "genes.csv", index=False)
    for a in ARMS:
        pd.DataFrame({"colID": np.arange(1, len(probs[a]) + 1), "probability": probs[a]}).to_csv(extra / f"{a}.csv", index=False)
    (outdir / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    return probs[pick], pick


def write_submission(prob, path):
    sub = pd.DataFrame({"colID": np.arange(1, 501), "probability": np.clip(prob, 0.0, 1.0)})
    assert len(sub) == 500 and sub["colID"].tolist() == list(range(1, 501))
    assert np.isfinite(sub["probability"]).all() and sub["probability"].between(0, 1).all()
    sub.to_csv(path, index=False)


def main():
    if WORK.exists():
        shutil.rmtree(WORK)
    WORK.mkdir(parents=True)
    sys.stdout = Tee(WORK / "log.txt")
    try:
        log("smoke run")
        run(SMOKE, WORK / "smoke")
        log("smoke ok; full run")
        prob, pick = run(FULL, WORK / "full")
        write_submission(prob, OUT / "submission.csv")
        log(f"SUBMIT: {OUT / 'submission.csv'} (arm {pick}, mean p {prob.mean():.4f}, expected positives {prob.sum():.1f})")
    except Exception:
        traceback.print_exc(file=sys.stdout)
        raise
    finally:
        sys.stdout.flush()
        with zipfile.ZipFile(OUT / "results.zip", "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(WORK.rglob("*")):
                if f.is_file():
                    z.write(f, f.relative_to(WORK))
        sys.stdout = sys.stdout.out


if __name__ == "__main__":
    main()
