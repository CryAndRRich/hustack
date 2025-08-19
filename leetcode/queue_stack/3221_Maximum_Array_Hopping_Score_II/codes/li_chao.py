from typing import List

class LiChao:
    def __init__(self, l, r):
        self.l = l; 
        self.r = r
        self.line = None
        self.left = None
        self.right = None
    def eval(self, line, x):
        return line[0] * x + line[1]
    def add(self, new, l=None, r=None):
        if l is None:
            l = self.l; r = self.r
        if self.line is None:
            self.line = new
            return
        mid = (l + r) >> 1
        if self.eval(new, mid) > self.eval(self.line, mid):
            self.line, new = new, self.line
        if l == r:
            return
        if self.eval(new, l) > self.eval(self.line, l):
            if self.left is None:
                self.left = LiChao(l, mid)
            self.left.add(new, l, mid)
        elif self.eval(new, r) > self.eval(self.line, r):
            if self.right is None:
                self.right = LiChao(mid + 1, r)
            self.right.add(new, mid + 1, r)
    def query(self, x, l=None, r=None):
        if l is None:
            l = self.l; r = self.r
        if self.line is None:
            return -10**30
        res = self.eval(self.line, x)
        if l == r:
            return res
        mid = (l + r) >> 1
        if x <= mid:
            if self.left is None:
                return res
            return max(res, self.left.query(x, l, mid))
        else:
            if self.right is None:
                return res
            return max(res, self.right.query(x, mid + 1, r))

class Solution:
    def maxScore(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        root = LiChao(0, n - 1)
        dp = [0] * n
        root.add((-nums[n - 1], (n - 1) * nums[n - 1] + dp[n - 1]))
        for i in range(n-2, -1, -1):
            best = root.query(i)
            dp[i] = best
            root.add((-nums[i], i * nums[i] + dp[i]))
        return dp[0]
