from typing import List

class Solution:
    def totalReplacements(self, ranks: List[int]) -> int:
        best = ranks[0]
        cnt = 0
        for r in ranks[1:]:
            if r < best:
                best = r
                cnt += 1
        return cnt
