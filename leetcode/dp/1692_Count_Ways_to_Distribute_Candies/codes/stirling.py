MOD = 10**9 + 7

def build_fact_and_ifact(n, mod):
    fact = [1] * (n + 1)
    for i in range(1, n + 1):
        fact[i] = fact[i - 1] * i % mod
    ifact = [1] * (n + 1)
    ifact[n] = pow(fact[n], mod - 2, mod)
    for i in range(n, 0, -1):
        ifact[i - 1] = ifact[i] * i % mod
    return fact, ifact

def nCk_mod(n, k, fact, ifact, mod):
    if k < 0 or k > n:
        return 0
    return fact[n] * ifact[k] % mod * ifact[n - k] % mod

def stirling2_mod(n, k, mod):
    if k < 0 or k > n:
        return 0
    fact, ifact = build_fact_and_ifact(k, mod)
    fact_k = fact[k]
    total = 0
    for m in range(k + 1):
        binom = nCk_mod(k, m, fact, ifact, mod)
        term = binom * pow(k - m, n, mod) % mod
        if m & 1:
            total = (total - term) % mod
        else:
            total = (total + term) % mod
    inv_fact_k = pow(fact_k, mod - 2, mod)
    return total * inv_fact_k % mod

class Solution:
    def waysToDistribute(self, n: int, k: int) -> int:
        return stirling2_mod(n, k, MOD)
