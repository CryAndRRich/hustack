from typing import List

MOD = 998244353
PRIMITIVE_ROOT = 3

class Solution:
    def mod_pow(self, a: int, b: int, mod: int) -> int:
        res = 1
        while b:
            if b & 1:
                res = res * a % mod
            a = a * a % mod
            b >>= 1
        return res

    def ntt(self, a: List[int], invert: bool) -> List[int]:
        n = len(a)
        j = 0
        for i in range(1, n):
            bit = n >> 1
            while j & bit:
                j ^= bit
                bit >>= 1
            j ^= bit
            if i < j:
                a[i], a[j] = a[j], a[i]

        len_ = 2
        while len_ <= n:
            wlen = self.mod_pow(PRIMITIVE_ROOT, (MOD - 1) // len_, MOD)
            if invert:
                wlen = self.mod_pow(wlen, MOD - 2, MOD)
            for i in range(0, n, len_):
                w = 1
                for j in range(len_ // 2):
                    u = a[i + j]
                    v = a[i + j + len_ // 2] * w % MOD
                    a[i + j] = (u + v) % MOD
                    a[i + j + len_ // 2] = (u - v + MOD) % MOD
                    w = w * wlen % MOD
            len_ <<= 1

        if invert:
            inv_n = self.mod_pow(n, MOD - 2, MOD)
            for i in range(n):
                a[i] = a[i] * inv_n % MOD

        return a

    def multiply(self, poly1: List[int], poly2: List[int]) -> List[int]:
        n = 1
        while n < len(poly1) + len(poly2) - 1:
            n <<= 1

        a = [(x % MOD + MOD) % MOD for x in poly1] + [0] * (n - len(poly1))
        b = [(x % MOD + MOD) % MOD for x in poly2] + [0] * (n - len(poly2))

        a = self.ntt(a, invert=False)
        b = self.ntt(b, invert=False)
        for i in range(n):
            a[i] = a[i] * b[i] % MOD
        a = self.ntt(a, invert=True)

        result = []
        for x in a[:len(poly1) + len(poly2) - 1]:
            if x > MOD // 2:
                x -= MOD
            result.append(x)

        return result
