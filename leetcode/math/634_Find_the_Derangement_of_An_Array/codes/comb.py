MOD = 10**9 + 7

class Solution:
    def findDerangement(self, n: int) -> int:
        if n == 1:
            return 0
        
        fact = [1] * (n + 1)
        inv_fact = [1] * (n + 1)

        for i in range(1, n + 1):
            fact[i] = fact[i - 1] * i % MOD

        inv_fact[n] = pow(fact[n], MOD - 2, MOD)
        for i in range(n - 1, 0, -1):
            inv_fact[i] = inv_fact[i + 1] * (i + 1) % MOD

        derangement = 0
        sign = 1
        for k in range(n + 1):
            term = sign * inv_fact[k] % MOD
            derangement = (derangement + term) % MOD
            sign *= -1

        result = fact[n] * derangement % MOD
        return result
