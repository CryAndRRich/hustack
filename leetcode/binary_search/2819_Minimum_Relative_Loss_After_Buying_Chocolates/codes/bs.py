from typing import List

class Solution:
    def minimumRelativeLosses(self, prices: List[int], queries: List[List[int]]) -> List[int]:
        prices.sort()
        n = len(prices)
        sum_arr = [0] * (n + 1)
        for i in range(1, n + 1):
            sum_arr[i] = sum_arr[i - 1] + prices[i - 1]

        m = len(queries)
        ind = list(range(m))
        ind.sort(key=lambda x: queries[x][0])
        j = 0
        result = [0] * m

        for i in ind:
            p, num = queries[i]
            pp = p << 1
            while j < n and prices[j] <= p:
                j += 1
            num1 = min(num, j)
            left, right = 1, min(num1, n - num)
            while left <= right:
                mid = (left + right) >> 1
                if prices[num1 - mid] > pp - prices[n - (num - num1) - mid]:
                    left = mid + 1
                else:
                    right = mid - 1
            num1 -= left - 1
            num2 = num - num1
            result[i] = num2 * pp - (sum_arr[n] - sum_arr[n - num2]) + sum_arr[num1]

        return result