from typing import List
MOD = 10**9 + 7

class Solution:
    def buildWall(self, height: int, width: int, bricks: List[int]) -> int:
        rows = []
        def gen(pos, mask):
            if pos == width:
                rows.append(mask)
                return
            for b in bricks:
                if pos + b <= width:
                    if pos + b < width:
                        gen(pos + b, mask | (1 << (pos + b)))
                    else:
                        gen(pos + b, mask)
        gen(0, 0)
        compat = {m: [] for m in rows}
        for a in rows:
            for b in rows:
                if a & b == 0:
                    compat[a].append(b)
        dp = {m: 1 for m in rows}
        for _ in range(height - 1):
            ndp = {m: 0 for m in rows}
            for m in rows:
                for nxt in compat[m]:
                    ndp[nxt] = (ndp[nxt] + dp[m]) % MOD
            dp = ndp
        return sum(dp.values()) % MOD
