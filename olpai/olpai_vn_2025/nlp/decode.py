from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

from nlp.data import Group
from nlp.metric import edge_counts, total

GRID = {
    "a": (0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0),
    "thr": (-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0),
    "b21": (-2.0, -1.5, -1.0, -0.5, 0.0, 0.5),
    "b12": (-2.0, -1.5, -1.0, -0.5, 0.0, 0.5),
    "w_name": (0.0, 0.25, 0.5, 1.0),
    "w_len": (0.0, 0.25, 0.5, 1.0),
    ("tail_frac", "tail_pen"): ((0.0, 0.0),) + tuple((f, p) for f in (0.005, 0.01, 0.02, 0.04) for p in (0.5, 1.0, 2.0, 4.0)),
}
DEFAULT = {"a": 0.5, "thr": 1.0, "b21": -0.5, "b12": -0.5, "w_name": 0.0, "w_len": 0.0, "tail_frac": 0.0, "tail_pen": 0.0}
LEN_RATIO = math.log(1.27)

NAME_RE = re.compile(r"\b[A-Z][A-Za-z'-]+\b|\d+")
STOP = set("""A About After All Also An And Any Are As At Be Because Before But By Can Could Did Do Does Don Even Every
For From Had Has Have He Her Here Hers Him His How However I If In Is It Its Just Let Like Maybe Me My No Nor Not Now Of Oh
On Once Only Or Our Out Please Really Right See She Should Since So Some Still Such Than That The Their Them Then There
These They This Those Though Thus To Too Um Uh Unlike Until Up Very Was We Well Were What When Where Whether Which While
Who Why Will With Without Would Yes Yet You Your""".split())


@dataclass
class Candidate:
    kind: str
    en: tuple[int, ...]
    vi: tuple[int, ...]
    bi: float
    rr: float
    name: float = 0.0
    length: float = 0.0
    tail: float = 1.0
    z_rr: float = 0.0
    z_bi: float = 0.0


def en_names(text: str) -> set[str]:
    return {t for t in NAME_RE.findall(text) if t not in STOP}


def load_candidates(path: Path, en_text: dict[int, str], vi_text: dict[int, str], en_order: list[int]) -> list[Candidate]:
    position = {e: i for i, e in enumerate(en_order)}
    n = len(en_order)
    out = []
    with open(path, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            en = tuple(int(x) for x in r["en"].split(";"))
            vi = tuple(int(x) for x in r["vi"].split(";"))
            en_s = " ".join(en_text[e] for e in en)
            vi_s = " ".join(vi_text[v] for v in vi)
            names = en_names(en_s)
            vi_tokens = set(re.findall(r"\w+", vi_s))
            ratio = math.log((len(vi_s.split()) + 1) / (len(en_s.split()) + 1))
            out.append(Candidate(r["kind"], en, vi, float(r["bi"]), float(r.get("rr") or 0.0),
                                 name=1.0 if names & vi_tokens else 0.0, length=abs(ratio - LEN_RATIO),
                                 tail=min((n - 1 - position[e]) / n for e in en)))
    _standardize(out)
    return out


def _standardize(cands: list[Candidate]) -> None:
    ref = [c for c in cands if c.kind == "11"] or cands
    for key in ("rr", "bi"):
        values = [getattr(c, key) for c in ref]
        mean = sum(values) / len(values)
        std = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
        for c in cands:
            setattr(c, f"z_{key}", (getattr(c, key) - mean) / std if std > 1e-9 else 0.0)


def candidate_score(c: Candidate, p: dict) -> float:
    s = p["a"] * c.z_rr + (1 - p["a"]) * c.z_bi + p["w_name"] * c.name - p["w_len"] * c.length
    if c.kind == "21":
        s += p["b21"]
    elif c.kind == "12":
        s += p["b12"]
    if c.tail < p["tail_frac"]:
        s -= p["tail_pen"]
    return s


def decode(cands: list[Candidate], p: dict, en_ids: list[int], vi_ids: list[int]) -> list[Group]:
    scored = sorted(((candidate_score(c, p), i) for i, c in enumerate(cands)), reverse=True)
    used_en, used_vi, groups = set(), set(), []
    thr = p["thr"]
    for s, i in scored:
        if s < thr:
            break
        c = cands[i]
        if used_en.isdisjoint(c.en) and used_vi.isdisjoint(c.vi):
            used_en.update(c.en)
            used_vi.update(c.vi)
            groups.append((c.en, c.vi))
    groups += [((e,), ()) for e in en_ids if e not in used_en]
    groups += [((), (v,)) for v in vi_ids if v not in used_vi]
    return groups


def evaluate(pools: list[dict], p: dict) -> dict:
    return total([edge_counts(x["gold"], decode(x["cands"], p, x["en"], x["vi"])) for x in pools])


def tune(pools: list[dict], fixed: dict | None = None, passes: int = 3) -> tuple[dict, dict]:
    fixed = fixed or {}
    p = {**DEFAULT, **fixed}
    best = evaluate(pools, p)
    for _ in range(passes):
        improved = False
        for key, values in GRID.items():
            names = key if isinstance(key, tuple) else (key,)
            if any(name in fixed for name in names):
                continue
            for value in values:
                update = dict(zip(names, value if isinstance(key, tuple) else (value,)))
                if all(p[name] == v for name, v in update.items()):
                    continue
                trial = {**p, **update}
                result = evaluate(pools, trial)
                if result["f1"] > best["f1"] + 1e-9:
                    p, best, improved = trial, result, True
        if not improved:
            break
    return p, best
