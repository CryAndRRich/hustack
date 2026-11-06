from pathlib import Path

import numpy as np
import pandas as pd

COUNT_CAP = 2000
SHRINK_A = 40
POSITION = {"GOALKEEPER": "GK", "DEFENDER": "DEF", "MIDFIELDER": "MID", "FORWARD": "FWD"}
STRICT = {"lenient": 0, "average": 1, "strict": 2, "very strict": 3}
CATS = ["position"]
FEATURES = ["strict", "position", "home", "cup", "shy", "ypr", "yel_ff", "starts", "yel_missing", "yel_last1",
            "yel_last5", "yel_last10", "yel_last20", "rest_days", "dow"]

def find(name):
    root = Path("/kaggle/input")
    hits = sorted(root.rglob(name)) if root.exists() else []
    if hits:
        return hits[0].parent
    try:
        return Path(__file__).resolve().parent.parent / "data"
    except NameError:
        return Path.cwd() / "data"

def norm_id(s):
    return s.str.strip().str.upper()

def read(d, name):
    return pd.read_csv(d / name, dtype=str, keep_default_na=False)

def count(s):
    x = pd.to_numeric(s, errors="coerce")
    return x.where((x >= 0) & (x < COUNT_CAP))

def load_rows():
    d = find("matches.csv")
    tr, te, m, p, rf = (read(d, f) for f in ("train.csv", "test.csv", "matches.csv", "players.csv", "referees.csv"))
    m["k"] = pd.to_datetime(m.kickoff_time.str.strip(), format="mixed", utc=True)
    assert m.k.notna().all() and m.match_id.is_unique
    for c in ["referee_id", "home_club_id", "away_club_id"]:
        m[c] = norm_id(m[c])
    m["cup"] = (m.competition.str.strip().str.lower() == "cup").astype(int)
    p["club"] = norm_id(p.club_id)
    pos = norm_id(p.position)
    p["position"] = pos.map(POSITION).fillna(pos)
    assert set(p.position) <= {"GK", "DEF", "MID", "FWD"}, set(p.position)
    rf["referee_id"] = norm_id(rf.referee_id)
    rf["strict"] = rf.strictness_category.str.lower().str.split().str.join(" ").map(STRICT)
    assert rf.strict.notna().all() and rf.referee_id.is_unique
    tr["y"] = tr.pop("red_card").astype(float)
    te["y"] = np.nan
    a = pd.concat([tr, te], ignore_index=True)
    n = len(a)
    a = (a.merge(m[["match_id", "k", "referee_id", "home_club_id", "away_club_id", "cup"]], on="match_id")
          .merge(p[["player_id", "club", "position"]], on="player_id").merge(rf[["referee_id", "strict"]], on="referee_id"))
    assert len(a) == n
    print(f"rows {n}: train {tr.shape[0]} (red rate {tr.y.mean():.4f}), test {te.shape[0]}")
    return a, te.player_match_id.to_numpy()

def player_features(a):
    a["yel"], a["starts"] = count(a.prior_yellows), count(a.prior_starts)
    a["yel_missing"] = a.yel.isna().astype(int)
    a = a.sort_values(["player_id", "k"]).reset_index(drop=True)
    g = a.groupby("player_id")
    rows = g.cumcount()

    a["starts"] = ((a.starts - rows).groupby(a.player_id).ffill() + rows).fillna(rows)
    a["yel_ff"] = a.yel.groupby(a.player_id).ffill()
    a["ypr"] = a.yel_ff / (a.starts + 1)
    mu = {p: d.yel_ff.sum() / d.starts.sum() for p, d in a.groupby("position")}
    prior = a.position.map(mu).astype(float)
    a["shy"] = (a.yel_ff.fillna(prior * a.starts) + SHRINK_A * prior) / (a.starts + SHRINK_A)
    for w in (1, 5, 10, 20):
        a[f"yel_last{w}"] = a.yel - g.yel.shift(w)
    a["rest_days"] = (a.k - g.k.shift(1)).dt.days
    a["dow"] = a.k.dt.dayofweek
    a["home"] = np.select([a.club == a.home_club_id, a.club == a.away_club_id], [1, 0], -1)
    return a

def load():
    a, test_ids = load_rows()
    a = player_features(a)
    a["position"] = a.position.astype("category")
    print(f"features {len(FEATURES)}")
    return a, test_ids
