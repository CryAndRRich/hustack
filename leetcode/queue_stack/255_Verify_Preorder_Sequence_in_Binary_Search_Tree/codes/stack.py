from typing import List

class Solution:
    def verifyPreorder(self, preorder: List[int]) -> bool:
        stack = []
        lower_bound = float("-inf")
        
        for value in preorder:
            if value < lower_bound:
                return False
            
            while stack and value > stack[-1]:
                lower_bound = stack.pop()
            
            stack.append(value)
        
        return True
