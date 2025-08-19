from collections import deque
from typing import List

class Solution:
    def kEmptySlots(self, flowers: List[int], k: int) -> int:
        if not flowers or k + 2 > len(flowers):
            return -1
        n = len(flowers)
        days = [0] * n
        for i, pos in enumerate(flowers):
            days[pos - 1] = i + 1

        res = float('inf')
        dayMin = deque()

        for i in range(1, k + 1):
            while dayMin and days[dayMin[-1]] > days[i]:
                dayMin.pop()
            dayMin.append(i)

        for i in range(k + 1, n):
            maxDay = max(days[i - k - 1], days[i])
            if k == 0:
                res = min(res, maxDay)
            elif maxDay < days[dayMin[0]]:
                res = min(res, maxDay)

            if dayMin and dayMin[0] == i - k:
                dayMin.popleft()
            while dayMin and days[dayMin[-1]] > days[i]:
                dayMin.pop()
            dayMin.append(i)

        return -1 if res == float('inf') else res
