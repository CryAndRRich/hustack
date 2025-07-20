from typing import List

class Solution:
    def visibleMountains(self, peaks: List[List[int]]) -> int:
        peaks = sorted(peaks, key = lambda peak: peak[0])

        stack = []
        end = float("-inf")

        for peak in peaks:
            left = peak[0] - peak[1]
            right = peak[0] + peak[1]

            while stack and stack[-1] >= left:
                stack.pop()

            if right > end:
                stack.append(left)
                end = right

        return len(stack)
        