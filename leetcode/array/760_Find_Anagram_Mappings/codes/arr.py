from collections import defaultdict
from typing import List

class Solution:
    def anagramMappings(self, nums1: List[int], nums2: List[int]) -> List[int]:
        index_map = defaultdict(list)
        for i, val in enumerate(nums2):
            index_map[val].append(i)
        
        result = []
        for val in nums1:
            result.append(index_map[val].pop())
        
        return result
