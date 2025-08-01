class Solution:
    def isArmstrong(self, n: int) -> bool:
        digits = list(map(int, str(n)))
        k = len(digits)
        return sum(d ** k for d in digits) == n
