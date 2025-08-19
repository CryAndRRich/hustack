from typing import List
import bisect

class Solution:
    def minOperations(self, nums: List[int]) -> int:
        tails = []
        for x in nums:
            i = bisect.bisect_right(tails, -x)
            if i == len(tails):
                tails.append(-x)
            else:
                tails[i] = -x
        return len(tails)
