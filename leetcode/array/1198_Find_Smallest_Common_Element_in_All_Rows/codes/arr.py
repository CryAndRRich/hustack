from collections import Counter
from typing import List

class Solution:
    def smallestCommonElement(self, mat: List[List[int]]) -> int:
        count = Counter()
        m = len(mat)
        
        for row in mat:
            for num in row:
                count[num] += 1
        
        for num in sorted(count):
            if count[num] == m:
                return num
        return -1
