from typing import List

class Solution:
    def check(self, timeReq: List[int], maxTime: int, splitTime: int) -> bool:
        currentWBC = 1
        curTime = 0
        n = len(timeReq)

        for i in range(n):
            if currentWBC == 0:
                return False
            while curTime + splitTime + timeReq[i] <= maxTime and currentWBC < n - i:
                currentWBC <<= 1
                curTime += splitTime
            currentWBC -= 1
        return True

    def minEliminationTime(self, timeReq: List[int], splitTime: int) -> int:
        timeReq.sort(reverse=True)
        left, right = timeReq[0], timeReq[0] + 20 * splitTime

        while left < right:
            mid = (left + right) // 2
            if self.check(timeReq, mid, splitTime):
                right = mid
            else:
                left = mid + 1
        return left
