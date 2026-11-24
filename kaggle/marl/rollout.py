import random

from marl1d.bots import HeuristicBot, TacticalBot
from marl1d.engine import Engine

BOTS = {"tac": TacticalBot, "heu": HeuristicBot}
STATES = {}


def play(job):
    table, index, variant, n, seed = job
    rng = random.Random(seed)
    start = STATES[table][index]
    bot = BOTS[variant]
    w1 = w2 = draws = turns = 0
    for _ in range(n):
        s = start.clone()
        b0, b1 = bot(seed=rng.randrange(1 << 30)), bot(seed=rng.randrange(1 << 30))
        while not s.is_game_over and s.turn <= s.max_turns:
            s, _ = Engine.step(s, b0.select_action(0, s), b1.select_action(1, s))
        w1 += s.winner == 0
        w2 += s.winner == 1
        draws += s.winner not in (0, 1)
        turns += s.turn
    return table, index, variant, (w1, w2, draws, n, turns)
