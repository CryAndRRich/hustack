import csv

import numpy as np

import runtime
from clean import prepare, template_db, true_nos, v1_wiener
from data import TRAINING, load_challenge, load_training
from metric import SPLITS, WINDOW, WINDOW_STARTS, scores, window_index
from runtime import log
from tilt import detilt_modes

LOUD = [3, 10]
COMPONENTS = ["nos", "leak", "ig", "tilt"]


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def make_case(name, ch, land, compl, nos):
    return {"name": name, "ch": ch, "land": land, "compl": compl, "nos": nos, **prepare(ch)}


def louder(case, k):
    ch, land = dict(case["ch"]), {key: k * v for key, v in case["land"].items()}
    for key in ("LH1", "LH2", "LHZ"):
        ch[key] = case["ch"][key] + (k - 1) * case["land"][key]
    return make_case(f"{case['name']}x{k}", ch, land, case["compl"], case["nos"])


def errors(case, u_raw):
    land = case["land"]
    e = {"nos": (1 - case["a"] - case["b"]) * case["nos"], "leak": -(case["a"] * land["LH1"] + case["b"] * land["LH2"]),
         "ig": case["compl"] - case["igz"]}
    e["tilt"] = u_raw - land["LHZ"] - sum(e.values())
    return e


def write_submission(path, compl, eq):
    rows = [(f"{w}_{i}", float(compl[i]), float(eq[i]))
            for w in SPLITS["all"] for i in range(WINDOW_STARTS[w], WINDOW_STARTS[w] + WINDOW)]
    vals = np.array([r[1:] for r in rows])
    assert len(rows) == 6000 and len({r[0] for r in rows}) == 6000 and np.isfinite(vals).all()
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "Z_compliance", "Z_earthquake"])
        w.writerows(rows)


def eq_arms(case, template):
    ch = case["ch"]
    for mode, u in detilt_modes(ch["LHZ"], ch["LH1"], ch["LH2"], case["a"], case["b"], case["ang"], case["igz"]):
        if mode == "raw":
            yield "none", u
        yield mode + "+v1", v1_wiener(u, template)
        yield mode + "+v1pad", v1_wiener(u, template, pad=True)


def run():
    try:
        root = runtime.data_root()
        log("data root", root)
        cases, groups = [], {"base": [], "MAIN": [], **{f"x{k}": [] for k in LOUD}}
        for name in TRAINING:
            s = load_training(root, name)
            base = make_case(name, s["ch"], s["land"], s["compl"], true_nos(s["ch"], s["land"], s["compl"]))
            groups["base"].append(len(cases))
            if name == "MAIN":
                groups["MAIN"].append(len(cases))
            cases.append(base)
            for k in LOUD:
                groups[f"x{k}"].append(len(cases))
                cases.append(louder(base, k))
        log("cases", len(cases))

        def u_raw(cs):
            ch = cs["ch"]
            return ch["LHZ"] - cs["a"] * ch["LH1"] - cs["b"] * ch["LH2"] - cs["igz"]

        template = np.median([template_db(u_raw(cases[i]) - cases[i]["land"]["LHZ"]) for i in groups["base"]], axis=0)
        w = window_index("all")
        table, comp_rms, win_err = {}, {c: [] for c in COMPONENTS + ["v1_err", "v1pad_err"]}, {"v1": [], "v1pad": []}
        for j, cs in enumerate(cases):
            z, lz = cs["ch"]["LHZ"], cs["land"]["LHZ"]
            for arm, y in eq_arms(cs, template):
                table.setdefault(arm, []).append(scores(y, lz, z))
            u = u_raw(cs)
            err = errors(cs, u)
            diag = {"diag_no_leak": v1_wiener(u - err["leak"], template, pad=True),
                    "diag_no_ig": v1_wiener(u - err["ig"], template, pad=True),
                    "diag_no_tilt": v1_wiener(u - err["tilt"], template, pad=True),
                    "diag_no_leak_ig_tilt": v1_wiener(lz + err["nos"], template, pad=True),
                    "diag_no_nos_no_wiener": u - err["nos"]}
            for arm, y in diag.items():
                table.setdefault(arm, []).append(scores(y, lz, z))
            for c in COMPONENTS:
                comp_rms[c].append(rms(err[c][w]))
            for tag, pad in (("v1", False), ("v1pad", True)):
                y = v1_wiener(u, template, pad=pad) - lz
                comp_rms[tag + "_err"].append(rms(y[w]))
                win_err[tag].append([rms(y[s0:s0 + WINDOW]) for s0 in WINDOW_STARTS])
            if j % 60 == 0:
                log("scored", j, "/", len(cases), cs["name"])

        summary = {}
        for arm, rows in table.items():
            sm = {g: float(np.mean([rows[i]["all"] for i in idx])) for g, idx in groups.items()}
            sm["MAIN_full"] = rows[groups["MAIN"][0]]["full"]
            sm["crit"] = float(np.mean([sm["base"], sm["x3"], sm["x10"]]))
            summary[arm] = sm
        ranked = sorted((a for a in summary if not a.startswith("diag")), key=lambda a: -summary[a]["crit"])
        for arm in ranked + sorted(a for a in summary if a.startswith("diag")):
            log("EQ %-22s" % arm, " ".join("%s %.2f" % kv for kv in summary[arm].items()))
        for g, idx in groups.items():
            log("window rms by component", g, {c: round(float(np.median([comp_rms[c][i] for i in idx])), 1)
                                                for c in comp_rms})
            for tag in win_err:
                log("  per-window error", tag, np.round(np.median([win_err[tag][i] for i in idx], axis=0), 0).tolist())
        fam_best = {}
        for a in ranked:
            fam_best.setdefault(a.split("_")[0].split("+")[0], a)
        chosen = ranked[0]
        log("family best:", fam_best, "| EQ chosen", chosen)
        log("REGRESSION vs V1: raw+v1 base %.2f (V1 90.48), none base %.2f (V1 71.84)" % (
            summary["raw+v1"]["base"], summary["none"]["base"]))

        ch = load_challenge(root, "SYNTH")["ch"]
        assert len(ch["LHZ"]) == 864000, len(ch["LHZ"])
        sp = {"ch": ch, **prepare(ch)}
        wanted = set(fam_best.values()) | {chosen}
        extra = runtime.WORK / "extra"
        extra.mkdir(parents=True, exist_ok=True)
        sub = runtime.out_dir() / "submission.csv"
        ref = None
        for arm, y in eq_arms(sp, template):
            if arm == "none":
                ref = y
            if arm in wanted:
                write_submission(extra / f"submission_{arm}.csv", sp["igz"], y)
                log("SYNTH arm %-22s rms eq %.1f, removed vs none %.1f" % (arm, rms(y), rms(y - ref)))
            if arm == chosen:
                write_submission(sub, sp["igz"], y)
        runtime.save_json({"chosen": chosen, "family_best": fam_best, "summary": summary,
                           "per_case": {cs["name"]: {a: table[a][j] for a in wanted | {"raw+v1"}}
                                        for j, cs in enumerate(cases)},
                           "component_rms": {cs["name"]: {c: comp_rms[c][j] for c in comp_rms}
                                             for j, cs in enumerate(cases)}}, "result.json")
        log("SUBMIT:", sub, "(compl V1 config, eq %s)" % chosen)
    finally:
        runtime.save_code()
        runtime.zip_results()


if __name__ == "__main__":
    run()
