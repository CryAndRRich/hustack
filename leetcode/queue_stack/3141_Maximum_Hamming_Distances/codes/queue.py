from collections import deque
from typing import List

class Solution:
    def maxHammingDistances(self, nums: List[int], m: int) -> List[int]:
        n = 1 << m
        dist = [-1] * n
        q = deque()
        for x in nums:
            dist[x] = 0
            q.append(x)
        while q:
            u = q.popleft()
            for b in range(m):
                v = u ^ (1 << b)
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    q.append(v)
        mask = n - 1
        ans = []
        for x in nums:
            ans.append(m - dist[mask ^ x])
        return ans
