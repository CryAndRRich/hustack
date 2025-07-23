from typing import List

class Solution:
    def minNumberOfPrimes(self, n: int, m: int) -> int:
        def generate_primes(limit_count: int) -> List[int]:
            size = 8000  
            is_prime = [True] * size
            is_prime[0] = is_prime[1] = False

            for i in range(2, int(size ** 0.5) + 1):
                if is_prime[i]:
                    for j in range(i * i, size, i):
                        is_prime[j] = False

            primes = [i for i, val in enumerate(is_prime) if val]
            return primes[:limit_count]

        primes = generate_primes(m)
        dp = [float("inf")] * (n + 1)
        dp[0] = 0

        for p in primes:
            for i in range(p, n + 1):
                if dp[i - p] + 1 < dp[i]:
                    dp[i] = dp[i - p] + 1

        return dp[n] if dp[n] != float("inf") else -1
