import math

import trueskill

RES = ["A", "B", "C", "D", "E"]
SKIP = "skip"
N = 6
EXTRA = [SKIP] * (N - len(RES))
COVER = RES + EXTRA
DETERMINISTIC = ("rotate0", "rotate", "radicaleq", "equalize", "caste", "turntaking")
HYPOTHESES = DETERMINISTIC + ("random3", "random", "unknown")
PRIOR = {h: (math.log(1e-3) if h == "unknown" else 0.0) for h in HYPOTHESES}


def round_wins(moves):
    return [int(m != SKIP and moves.count(m) == 1) for m in moves]


def _assign(order, res):
    return [s[1] for s in sorted(zip(order, res))]


class Match:

    def __init__(self):
        self.moves = []
        self.wins = [0] * N
        self.pop = {r: 0 for r in RES}
        self.skill = [trueskill.Rating(mu=25, sigma=0.5)] * N
        self.debt = [(N - p + 1) / 100000 for p in range(N)]
        self.viol = {"equalize": [0] * N, "caste": [0] * N, "turntaking": [0] * N}
        self.plan = {"radicaleq": COVER, "equalize": COVER, "caste": COVER, "turntaking": EXTRA + RES}

    @property
    def round(self):
        return len(self.moves) + 1

    def move(self, name, p):
        r = self.round
        if name == "rotate0":
            return COVER[p]
        if name == "rotate":
            return COVER[(p + r - 1) % N]
        return self.plan[name][p]

    def dist(self, name, p):
        if name in DETERMINISTIC:
            return {self.move(name, p): 1.0}
        if name == "random":
            return {m: 1 / len(COVER) for m in COVER}
        if self.round == 1:
            pool = RES if name == "random3" else COVER
            return {m: 1 / len(pool) for m in pool}
        prev = self.moves[-1]
        own = prev[p]
        if name == "unknown":
            d = {m: 0.5 / (len(COVER) - 1) for m in COVER}
            d[own] = 0.5
            return d
        if own == SKIP:
            return {}
        if round_wins(prev)[p]:
            return {own: 1.0}
        opts = [b for b in RES if b not in prev] or [b for b in RES if prev.count(b) == 1] or RES
        d = {b: (1 / 3) / len(opts) for b in opts}
        d[own] = d.get(own, 0.0) + 2 / 3
        return d

    def observe(self, moves):
        prev_plan = dict(self.plan)
        self.moves.append(list(moves))
        won = round_wins(moves)
        self.wins = [w + x for w, x in zip(self.wins, won)]
        r = self.round
        for p, m in enumerate(moves):
            if m in RES:
                self.pop[m] += 1 + (p + 1) / 10000
        self._favours(won)
        res_sorted = sorted(self.pop.items(), key=lambda it: it[1])
        names = [x[0] for x in res_sorted]
        by_wins = [p for _, p in sorted((self.wins[p], p) for p in range(N))]
        by_debt = [p for _, p in sorted((-self.debt[p], p) for p in range(N))]
        for name, decay in (("equalize", 0.7), ("caste", 0.7), ("turntaking", 0.6)):
            v = [x * decay for x in self.viol[name]]
            for p in range(N):
                if moves[p] != prev_plan[name][p]:
                    v[p] += 1
            self.viol[name] = v
        self.plan["radicaleq"] = _assign(by_wins, names + EXTRA)
        for name, order in (("equalize", by_wins), ("caste", by_debt)):
            learners = [p for p in order if self.viol[name][p] < 1]
            self.plan[name] = _assign(learners + [p for p in order if p not in learners], names + EXTRA)
        learners = [p for p in by_debt if self.viol["turntaking"][p] < 1]
        takers = [p for p in by_debt if p not in learners] + learners
        free = [x for x in res_sorted if x[1] < r * 1.45]
        ordered = [x[0] for x in [x for x in res_sorted if x not in free] + free]
        self.plan["turntaking"] = _assign(takers, EXTRA + ordered)

    def _favours(self, won):
        winners = [0 if w else 1 for w in won]
        env = trueskill.global_env()
        for me in range(N):
            mine = self.skill[me]
            for other in range(N):
                if me == other:
                    continue
                theirs = self.skill[other]
                denom = math.sqrt(2 * trueskill.BETA ** 2 + mine.sigma ** 2 + theirs.sigma ** 2)
                if mine.mu == theirs.mu:
                    p_win = 0.5
                elif mine.mu > theirs.mu:
                    p_win = env.cdf((mine.mu - theirs.mu) / denom)
                else:
                    p_win = 1 - env.cdf((theirs.mu - mine.mu) / denom)
                nondraw = 1 - trueskill.quality_1vs1(mine, theirs)
                if winners[me] == 1 or winners[other] == 0:
                    self.debt[me] -= p_win * nondraw
                if winners[me] == 0 or winners[other] == 1:
                    self.debt[me] += (1 - p_win) * nondraw
        self.skill = [s[0] for s in trueskill.rate([(r,) for r in self.skill], winners)]


class Tracker:

    def __init__(self, opponents):
        self.opp = list(opponents)
        self.state = Match()
        self.logp = dict(PRIOR)

    def observe(self, moves):
        for h in HYPOTHESES:
            if self.logp[h] == -math.inf:
                continue
            for p in self.opp:
                q = self.state.dist(h, p).get(moves[p], 0.0)
                self.logp[h] = self.logp[h] + math.log(q) if q > 0 else -math.inf
                if q == 0:
                    break
        self.state.observe(moves)

    def posterior(self):
        top = max(self.logp.values())
        w = {h: math.exp(v - top) for h, v in self.logp.items() if v > -math.inf}
        z = sum(w.values())
        return {h: x / z for h, x in w.items()}
