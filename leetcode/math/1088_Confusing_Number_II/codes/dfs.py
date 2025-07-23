class Solution:
    def confusingNumberII(self, n: int) -> int:
        rotate = {0: 0, 1: 1, 6: 9, 8: 8, 9: 6}
        digits = [0, 1, 6, 8, 9]
        self.count = 0

        def is_confusing(x: int) -> bool:
            original = x
            rotated = 0
            while x > 0:
                d = x % 10
                rotated = rotated * 10 + rotate[d]
                x //= 10
            return rotated != original

        def dfs(curr: int):
            if curr > n:
                return
            if curr != 0 and is_confusing(curr):
                self.count += 1
            for d in digits:
                if curr == 0 and d == 0:
                    continue
                dfs(curr * 10 + d)

        dfs(0)
        return self.count
