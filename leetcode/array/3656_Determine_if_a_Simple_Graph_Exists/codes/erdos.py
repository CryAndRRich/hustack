from typing import List
import bisect

class Solution:
    def simpleGraphExists(self, degrees: List[int]) -> bool:
        n = len(degrees)
        degrees.sort(reverse=True)
        
        total = sum(degrees)
        if total % 2 == 1:
            return False
        
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i+1] = prefix[i] + degrees[i]
        
        for k in range(1, n + 1):
            left = prefix[k]
            j = bisect.bisect_left(degrees, k, lo=0, hi=n, key=lambda x: -x)
            lo, hi = 0, n
            while lo < hi:
                mid = (lo + hi) // 2
                if degrees[mid] >= k:
                    lo = mid + 1
                else:
                    hi = mid
            j = lo
            
            right = k * (k - 1)
            right += k * max(0, j - k)
            right += prefix[n] - prefix[max(j, k)]
            
            if left > right:
                return False
        return True
