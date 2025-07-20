from typing import Optional

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def minimumFlips(self, root: Optional[TreeNode], result: bool) -> int:
        def dfs(node):
            if not node:
                return {True: float("inf"), False: float("inf")}

            if not node.left and not node.right: 
                val = bool(node.val)
                return {val: 0, not val: 1}

            if node.val == 5: 
                child = node.left or node.right
                child_dp = dfs(child)
                return {
                    True: child_dp[False],
                    False: child_dp[True]
                }

            left_dp = dfs(node.left)
            right_dp = dfs(node.right)

            dp = {True: float("inf"), False: float("inf")}
            for l_val in [False, True]:
                for r_val in [False, True]:
                    if node.val == 2:  
                        res = l_val or r_val
                    elif node.val == 3:  
                        res = l_val and r_val
                    elif node.val == 4:  
                        res = l_val ^ r_val
                    else:
                        continue  

                    cost = left_dp[l_val] + right_dp[r_val]
                    dp[res] = min(dp[res], cost)

            return dp

        dp = dfs(root)
        return dp[result]
