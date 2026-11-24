from marl1d.bots import HeuristicBot, TacticalBot
from marl1d.engine import Engine
from marl1d.roster import ROSTER_40

SIG_TO_CHAR = {v["sig"]: k for k, v in ROSTER_40.items()}
SIDES = (("player", "p1"), ("opp", "p2"))
WARMUP_BOTS = (("tac", TacticalBot), ("heu", HeuristicBot))
VISIBLE = ("hp", "pos", "en", "poise")


def character(row, side):
    name = row[side + "_char"]
    sig_char = SIG_TO_CHAR[row[side + "_loadout"].split("|")[0]]
    if isinstance(name, str):
        assert name == sig_char, (name, sig_char)
        return name
    return sig_char


def build(row):
    loadouts = {side: row[side + "_loadout"].split("|") for side, _ in SIDES}
    g = Engine.new_game(seed=0, lane_size=int(row["lane_size"]), max_turns=int(row["max_turns"]),
                        p1_char=character(row, "player"), p2_char=character(row, "opp"),
                        p1_loadout=loadouts["player"], p2_loadout=loadouts["opp"])
    g.turn = int(row["turn"])
    for side, attr in SIDES:
        p = getattr(g, attr)
        p.hp = int(row[side + "_hp"])
        p.max_hp = int(row[side + "_max_hp"])
        p.en = int(row[side + "_en"])
        p.max_en = int(row[side + "_max_en"])
        p.poise = int(row[side + "_poise"])
        p.poise_threshold = int(row[side + "_poise_threshold"])
        p.pos = int(row[side + "_pos"])
        p.staggered = bool(row[side + "_staggered"])
        p.hp_turn_start = p.hp
        p.pos_turn_start = p.pos
        assert p.loadout == loadouts[side]
    assert abs(g.p1.pos - g.p2.pos) == int(row["distance"])
    return g


def _visible(row, swapped):
    sides = ("opp", "player") if swapped else ("player", "opp")
    return tuple(int(row[s + "_" + f]) for s in sides for f in VISIBLE)


def replay(row, seed, swapped):
    chars = (character(row, "player"), character(row, "opp"))
    c1, c2 = (chars[1], chars[0]) if swapped else chars
    target, turn = _visible(row, swapped), int(row["turn"])
    for name, bot in WARMUP_BOTS:
        g = Engine.new_game(seed=seed, lane_size=int(row["lane_size"]), max_turns=int(row["max_turns"]),
                            p1_char=c1, p2_char=c2)
        a, b = bot(seed=seed + 1), bot(seed=seed + 2)
        while g.turn < turn and not g.is_game_over:
            g, _ = Engine.step(g, a.select_action(0, g), b.select_action(1, g))
        if g.turn == turn and not g.is_game_over and tuple(getattr(p, f) for p in (g.p1, g.p2) for f in VISIBLE) == target:
            p_row, o_row = (g.p2, g.p1) if swapped else (g.p1, g.p2)
            stag = p_row.staggered == bool(row["player_staggered"]) and o_row.staggered == bool(row["opp_staggered"])
            g.history = []
            return g, name, stag
    return None, None, None
