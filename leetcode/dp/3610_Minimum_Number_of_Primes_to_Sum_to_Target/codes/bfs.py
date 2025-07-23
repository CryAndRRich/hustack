from collections import deque
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
            primes = [i for i, v in enumerate(is_prime) if v]
            return primes[:limit_count]

        primes = generate_primes(m)
        visited = [False] * (n + 1)
        queue = deque([(0, 0)]) 
        visited[0] = True

        while queue:
            total, count = queue.popleft()
            if total == n:
                return count
            for p in primes:
                new_sum = total + p
                if new_sum <= n and not visited[new_sum]:
                    visited[new_sum] = True
                    queue.append((new_sum, count + 1))

        return -1
