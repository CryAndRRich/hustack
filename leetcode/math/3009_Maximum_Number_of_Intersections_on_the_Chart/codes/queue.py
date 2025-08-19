import heapq

class Solution:
    def maxIntersectionCount(self, y):
        res = 0
        candles = []
        for i in range(len(y) - 1):
            l = 2 * y[i] + (1 if i and y[i] < y[i + 1] else -1 if i else 0)
            r = 2 * y[i + 1]
            candles.append((min(l, r), max(l, r)))
        candles.sort()
        pq = []
        for l, r in candles:
            heapq.heappush(pq, r)
            while pq and pq[0] < l:
                heapq.heappop(pq)
            res = max(res, len(pq))
        return res
