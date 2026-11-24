import random

from marl1d.engine import Engine
from marl1d.roster import MOVES_100, ROSTER_40

SEED_MAX = 150000
STANDARD = [k for k in MOVES_100.keys() if k != "RST" and not (len(k) == 2 and k[0] in "IRVKHCU")]


def _key(char, loadout):
    pool = [m for m in STANDARD if m != ROSTER_40[char]["sig"]]
    return tuple(pool.index(m) for m in loadout[1:9])


def seed_index():
    n = len(STANDARD) - 1
    return {tuple(random.Random(s).sample(range(n), 8)): s for s in range(SEED_MAX)}


def find(index, chars, loadouts):
    for swapped in (False, True):
        first = 1 if swapped else 0
        s = index.get(_key(chars[first], loadouts[first]))
        if s is None:
            continue
        c1, c2 = (chars[1], chars[0]) if swapped else chars
        g = Engine.new_game(seed=s, p1_char=c1, p2_char=c2)
        l1, l2 = (loadouts[1], loadouts[0]) if swapped else loadouts
        if g.p1.loadout == l1 and g.p2.loadout == l2:
            return s, swapped
    return None, None
