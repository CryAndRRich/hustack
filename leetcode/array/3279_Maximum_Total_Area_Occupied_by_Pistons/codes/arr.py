from collections import defaultdict
from typing import List

class Solution:
    def maxArea(self, height: int, positions: List[int], directions: str) -> int:
        have = defaultdict(list)
        n = len(positions)
        s = sum(positions)
        a = 0
        directions = list(directions)
        for i in range(n):
            if directions[i] == 'U':
                a += 1
                have[height - positions[i]].append(i)
                have[height - positions[i] + height].append(i)
            else:
                a -= 1
                have[positions[i]].append(i)
                have[positions[i] + height].append(i)
        r = s
        now = 0
        for key in sorted(have.keys()):
            t = key - now
            s += t * a
            now = key
            r = max(r, s)
            for i in have[key]:
                if directions[i] == 'U':
                    directions[i] = 'D'
                    a -= 2
                else:
                    directions[i] = 'U'
                    a += 2
        return r
