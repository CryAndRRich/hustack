import heapq
from typing import List

class Solution:
    def connectSticks(self, sticks: List[int]) -> int:
        if len(sticks) <= 1:
            return 0

        heapq.heapify(sticks)
        cost = 0

        while len(sticks) > 1:
            first = heapq.heappop(sticks)
            second = heapq.heappop(sticks)
            new_stick = first + second
            cost += new_stick
            heapq.heappush(sticks, new_stick)

        return cost
