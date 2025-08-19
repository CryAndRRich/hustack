from typing import List

class Solution:
    def maxCoins(self, lane1: List[int], lane2: List[int]) -> int:
        n = len(lane1)
        NEG =  - 10 ** 30
        dp_prev = [[NEG] * 3 for _ in range(2)]
        ans = NEG
        for i in range(n):
            dp_curr = [[NEG] * 3 for _ in range(2)]
            for l in range(2):
                arr = lane1 if l == 0 else lane2
                for s in range(3):
                    best = NEG
                    start_s = 0 if l == 0 else 1
                    if s == start_s:
                        best = max(best, arr[i])
                    if dp_prev[l][s] != NEG:
                        best = max(best, dp_prev[l][s] + arr[i])
                    if s >= 1 and dp_prev[1 - l][s - 1] != NEG:
                        best = max(best, dp_prev[1 - l][s - 1] + arr[i])
                    dp_curr[l][s] = best
                    if best != NEG and best > ans:
                        ans = best
            dp_prev = dp_curr
        return ans
