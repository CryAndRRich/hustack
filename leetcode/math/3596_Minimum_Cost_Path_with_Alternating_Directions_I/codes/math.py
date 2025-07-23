class Solution:
    def minCost(self, m: int, n: int) -> int:
        # if m == 1 and n == 1:
        #     return 1
        # if m == 1 and n == 2:
        #     return 1 + (1 * 2) 
        # if m == 2 and n == 1:
        #     return 1 + (2 * 1) 
        # return -1

        return (m + n) * 2 - 3 if m + n <= 3 else -1
