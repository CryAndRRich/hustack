from typing import List

class Solution:
    def minimumTime(self, jobs: List[int], workers: List[int]) -> int:
        jobs.sort()
        workers.sort()

        def canFinish(days: int) -> bool:
            i = 0  
            j = 0  
            n = len(jobs)

            while i < n and j < n:
                if workers[j] * days >= jobs[i]:
                    i += 1 
                j += 1 
            return i == n 

        left, right = 1, max(jobs)
        while left < right:
            mid = (left + right) // 2
            if canFinish(mid):
                right = mid
            else:
                left = mid + 1
        return left
