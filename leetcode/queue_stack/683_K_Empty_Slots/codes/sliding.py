from typing import List

class Solution:
    def kEmptySlots(self, flowers: List[int], k: int) -> int:
        n = len(flowers)
        days = [0] * n
        for i, pos in enumerate(flowers):
            days[pos - 1] = i + 1

        left, right = 0, k + 1
        res = float('inf')

        for i in range(n):
            if right >= n:
                break
            if days[i] < days[left] or days[i] <= days[right]:
                if i == right:
                    res = min(res, max(days[left], days[right]))
                left = i
                right = i + k + 1

        return -1 if res == float('inf') else res
