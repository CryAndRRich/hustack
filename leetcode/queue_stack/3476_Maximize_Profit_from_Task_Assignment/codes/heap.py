from typing import List
import heapq
from collections import Counter

class Solution:
    def maxProfit(self, workers: List[int], tasks: List[List[int]]) -> int:
        res = 0
        additionalWorker = 1
        pq = [(-p, s) for s, p in tasks]
        heapq.heapify(pq)
        wrks = Counter(workers)  
        while pq:
            profit, skill = heapq.heappop(pq)
            profit = -profit
            if wrks[skill] > 0 or additionalWorker > 0:
                res += profit
                if wrks[skill] > 0:
                    wrks[skill] -= 1
                else:
                    additionalWorker = 0
        return res
