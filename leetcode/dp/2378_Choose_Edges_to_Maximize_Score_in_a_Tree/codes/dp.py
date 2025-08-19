from collections import defaultdict
from functools import lru_cache

class Solution:
    def maxScore(self, edges: List[List[int]]) -> int:
        n = len(edges)
        g = defaultdict(list)
        for i, (p, w) in enumerate(edges[1:], 1):
            g[p].append((i, w))

        @lru_cache(None)
        def dp(u, used):
            res = sum(dp(v, False) for v, _ in g[u])
            if not used:
                for v, w in g[u]:
                    if w <= 0: 
                        continue
                    cur = w + dp(v, True) + sum(dp(x, False) for x, _ in g[u] if x != v)
                    res = max(res, cur)
            return res
        
        return dp(0, False)
