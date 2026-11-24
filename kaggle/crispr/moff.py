import numpy as np
from config import SPACER_LEN
from moff_tables import M1_IME, M2_CE
COMPLEMENT = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}

def mutation_keys(spacer, candidate):
    positions, keys = ([], [])
    for i in range(SPACER_LEN):
        if spacer[i] == candidate[i]:
            continue
        pos = SPACER_LEN - i
        r = 'U' if spacer[i] == 'T' else spacer[i]
        d = COMPLEMENT.get(candidate[i], spacer[i])
        positions.append(pos)
        keys.append('p%dr%sd%s' % (pos, r, d))
    return (positions, keys)

def mde(keys):
    value = 1.0
    for k in keys:
        value *= M1_IME[k]
    return value

def ce(positions):
    n = len(positions)
    if n < 2:
        return 1.0
    product = 1.0
    for i in range(n):
        for j in range(i + 1, n):
            product *= M2_CE['%d&%d' % (positions[i], positions[j])]
    return product ** (2.0 / n)

def score(spacer, candidate):
    positions, keys = mutation_keys(spacer, candidate)
    return (mde(keys), ce(positions))

def columns(spacers, candidates):
    m, c = ([], [])
    for s, t in zip(spacers, candidates):
        a, b = score(s, t)
        m.append(a)
        c.append(b)
    m = np.asarray(m)
    c = np.asarray(c)
    return {'log_moff_mde': np.log(m + 1e-12), 'log_moff_ce': np.log(c + 1e-12), 'log_moff': np.log(m * c + 1e-12)}
