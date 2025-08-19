from typing import List
import bisect

class Fenwick:
    def __init__(self, n: int):
        self.n = n
        self.bit = [0] * (n + 1)
    def update(self, idx: int, delta: int):
        i = idx + 1
        while i <= self.n:
            self.bit[i] += delta
            i += i & -i
    def query(self, idx: int) -> int:
        if idx < 0: return 0
        i = idx + 1
        s = 0
        while i > 0:
            s += self.bit[i]
            i -= i & -i
        return s
    def range_query(self, l: int, r: int) -> int:
        if l > r: return 0
        return self.query(r) - self.query(l - 1)

class Solution:
    def minThreshold(self, nums: List[int], k: int) -> int:
        if not nums: return -1
        vals = sorted(set(nums))
        m = len(vals)

        def check(th: int) -> bool:
            fw = Fenwick(m)
            cnt = 0
            for num in nums:
                idx_l = bisect.bisect_right(vals, num)                 
                idx_r = bisect.bisect_right(vals, num + th) - 1       
                if idx_l <= idx_r:
                    cnt += fw.range_query(idx_l, idx_r)
                    if cnt >= k:
                        return True
                idx_num = bisect.bisect_left(vals, num)
                fw.update(idx_num, 1)
            return cnt >= k

        lo, hi = 0, max(nums) - min(nums)
        ans = -1
        while lo <= hi:
            mid = (lo + hi) // 2
            if check(mid):
                ans = mid
                hi = mid - 1
            else:
                lo = mid + 1
        return ans
