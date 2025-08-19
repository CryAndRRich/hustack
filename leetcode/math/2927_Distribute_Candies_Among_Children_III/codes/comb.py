class Solution:
    def distributeCandies(self, n: int, limit: int) -> int:
        def C2(x):
            return (x * (x - 1)) // 2 if x >= 2 else 0

        total = C2(n + 2)
        sub1 = 3 * C2(n - (limit + 1) + 2)
        add2 = 3 * C2(n - 2 * (limit + 1) + 2)
        sub3 = C2(n - 3 * (limit + 1) + 2)

        return total - sub1 + add2 - sub3
