from typing import List
import heapq

class Solution:
    def minEliminationTime(self, timeReq: List[int], splitTime: int) -> int:
        heapq.heapify(timeReq)

        while len(timeReq) > 1:
            t1, t2 = heapq.heappop(timeReq), heapq.heappop(timeReq)
            t2 += splitTime
            heapq.heappush(timeReq, t2) 

        return timeReq[0]