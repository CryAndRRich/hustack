from collections import defaultdict, deque
from typing import List

MOD = 10**9 + 7

class Solution:
    def queryConversions(self, conversions: List[List[int]], queries: List[List[int]]) -> List[int]:
        g = defaultdict(list)
        for a, b, f in conversions:
            g[a].append((b, f, 1))    
            g[b].append((a, f, 0))     
        n = len(conversions) + 1
        num = [0] * n
        den = [0] * n 
        num[0], den[0] = 1, 1

        vis = [False] * n
        vis[0] = True
        dq = deque([0])

        while dq:
            u = dq.popleft()
            for v, f, forward in g[u]:
                if not vis[v]:
                    if forward:
                        num[v] = num[u]
                        den[v] = (den[u] * f) % MOD
                    else:
                        num[v] = (num[u] * f) % MOD
                        den[v] = den[u]
                    vis[v] = True
                    dq.append(v)

        ans = []
        for a, b in queries:
            pa, qa = num[a], den[a]
            pb, qb = num[b], den[b]
            p = (pa * qb) % MOD
            q = (qa * pb) % MOD
            inv_q = pow(q, MOD - 2, MOD)
            ans.append((p * inv_q) % MOD)
        return ans
