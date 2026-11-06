import numpy as np
from scipy.stats import norm

N, M = 624, 397
BIG = 134217728


class RRNG:
    def __init__(self, seed):
        s = seed & 0xFFFFFFFF
        for _ in range(51):
            s = (69069 * s + 1) & 0xFFFFFFFF
        self.mt = []
        for _ in range(N):
            s = (69069 * s + 1) & 0xFFFFFFFF
            self.mt.append(s)
        self.mti = N

    def _twist(self):
        mt = self.mt
        for kk in range(N):
            y = (mt[kk] & 0x80000000) | (mt[(kk + 1) % N] & 0x7FFFFFFF)
            mt[kk] = mt[(kk + M) % N] ^ (y >> 1) ^ (0x9908B0DF if y & 1 else 0)
        self.mti = 0

    def unif(self):
        if self.mti >= N:
            self._twist()
        y = self.mt[self.mti]
        self.mti += 1
        y ^= y >> 11
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        y ^= y >> 18
        v = y * 2.3283064365386963e-10
        return min(max(v, 0.5 * 2.328306437080797e-10), 1 - 0.5 * 2.328306437080797e-10)

    def unif_index(self, dn):
        bits = int(np.ceil(np.log2(dn)))
        while True:
            v = 0
            for _ in range(0, bits + 1, 16):
                v = 65536 * v + int(self.unif() * 65536)
            v &= (1 << bits) - 1
            if v < dn:
                return v

    def sample(self, n, k):
        x, out = list(range(n)), []
        for _ in range(k):
            j = self.unif_index(n)
            out.append(x[j])
            n -= 1
            x[j] = x[n]
        return np.array(out)

    def rnorm(self, k):
        u = np.array([(int(BIG * self.unif()) + self.unif()) / BIG for _ in range(k)])
        return norm.ppf(u)
