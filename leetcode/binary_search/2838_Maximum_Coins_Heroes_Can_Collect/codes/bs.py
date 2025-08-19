from typing import List

class Solution:
    def maximumCoins(self, heroes: List[int], monsters: List[int], coins: List[int]) -> List[int]:
        paired = sorted(zip(monsters, coins))
        monsters_sorted = [m for m, _ in paired]
        prefix_sum = [0] * (len(coins) + 1)
        for i in range(len(coins)):
            prefix_sum[i + 1] = prefix_sum[i] + paired[i][1]

        def upper_bound(arr, x):
            left, right = 0, len(arr)
            while left < right:
                mid = (left + right) // 2
                if arr[mid] <= x:
                    left = mid + 1
                else:
                    right = mid
            return left

        result = []
        for h in heroes:
            idx = upper_bound(monsters_sorted, h)
            result.append(prefix_sum[idx])

        return result
