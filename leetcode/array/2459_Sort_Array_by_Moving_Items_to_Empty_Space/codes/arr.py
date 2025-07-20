from typing import List

class Solution:
    def sortArray(self, nums: List[int]) -> int:
        return min(self.getCountSorting(nums, 0), self.getCountSorting(nums, 1))

    def getCountSorting(self, nums: List[int], offset: int) -> int:
        count = 0
        for i in range(len(nums)):
            if nums[i] == 0 or i == nums[i] - offset:
                continue
            j = i
            count += 1
            while nums[j] > 0:
                nums[j] = -nums[j]
                j = -nums[j] - offset
                if j == i:
                    count += 1
                    break
        for i in range(len(nums)):
            nums[i] = abs(nums[i])
        return count
