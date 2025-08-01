from typing import List

class Solution:
    def smallestCommonElement(self, mat: List[List[int]]) -> int:
        def binary_search(arr, target):
            l, r = 0, len(arr) - 1
            while l <= r:
                mid = (l + r) // 2
                if arr[mid] == target:
                    return True
                elif arr[mid] < target:
                    l = mid + 1
                else:
                    r = mid - 1
            return False

        for num in mat[0]:
            if all(binary_search(row, num) for row in mat[1:]):
                return num
        return -1
