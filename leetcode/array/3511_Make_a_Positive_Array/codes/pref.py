from typing import List

class Solution:
    def makeArrayPositive(self, nums: List[int]) -> int:
        n = len(nums)
        pref = [0] * (n + 1)
        for i in range(n):
            pref[i+1] = pref[i] + nums[i]
        res = 0
        covered = -1
        for end in range(2, n):
            need = False
            for l in (3, 4, 5):
                start = end - (l - 1)
                if start < 0:
                    continue
                if covered >= start and covered <= end:
                    continue
                if pref[end+1] - pref[start] <= 0:
                    need = True
                    break
            if need:
                res += 1
                covered = end
        return res
