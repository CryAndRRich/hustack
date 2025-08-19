class Solution:
    def smallestFactorization(self, num: int) -> int:
        if num < 10:
            return num
        
        factors = []
        for d in range(9, 1, -1):
            while num % d == 0:
                factors.append(d)
                num //= d
        
        if num != 1:
            return 0
        
        factors.sort()
        result = 0
        for d in factors:
            result = result * 10 + d
            if result > 2**31 - 1:
                return 0
        return result
