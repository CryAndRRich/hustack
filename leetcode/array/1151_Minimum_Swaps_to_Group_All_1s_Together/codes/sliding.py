from typing import List

class Solution:
    def minSwaps(self, data: List[int]) -> int:
        ones = sum(data)
        if ones <= 1:
            return 0

        current_ones = sum(data[:ones])
        max_ones_in_window = current_ones

        for i in range(ones, len(data)):
            current_ones += data[i] - data[i - ones]
            max_ones_in_window = max(max_ones_in_window, current_ones)

        return ones - max_ones_in_window
