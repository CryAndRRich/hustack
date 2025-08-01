from typing import List

class Solution:
    def missingNumber(self, arr: List[int]) -> int:
        n = len(arr)
        diffs = [arr[i+1] - arr[i] for i in range(n-1)]
        d = min(diffs, key=abs) 

        for i in range(n - 1):
            if arr[i + 1] - arr[i] != d:
                return arr[i] + d
        return arr[0] 
