from typing import List
from sortedcontainers import SortedList

class Solution:
    def countOperationsToEmptyArray(self, nums: List[int]) -> int:
        N = len(nums)
        locs = {v: i for i, v in enumerate(nums)}          
        positions = SortedList(range(N))                

        ans = i = 0                                     
        for x in sorted(nums):                             
            r = locs[x]                                 
            j = positions.index(r)                      
            ans += (j - i) % len(positions)             
            positions.pop(j)                            
            i = j                                        
        return ans + N                                
