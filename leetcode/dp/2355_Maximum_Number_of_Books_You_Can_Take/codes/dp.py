from typing import List

class Solution:
    def maximumBooks(self, books: List[int]) -> int:
        n = len(books)
        dp = [0 for _ in range(n)]
        stack = []
        res = 0

        for i in range(n):
            while stack and books[stack[-1]] >= books[i] - (i - stack[-1]):
                stack.pop()

            if not stack:
                length = min(i + 1, books[i])
                first = books[i] - length + 1
                dp[i] = (books[i] + first) * length // 2
            else:
                j = stack[-1]
                length = i - j
                first = books[i] - length + 1
                dp[i] = dp[j] + (books[i] + first) * length // 2

            stack.append(i)
            res = max(res, dp[i])

        return res
