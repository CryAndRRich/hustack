from typing import List, Tuple
import math

class Solution:
    @staticmethod
    def beautifulPair(nums1: List[int], nums2: List[int]) -> List[int]:
        seen = {}
        n = len(nums1)
        res = (math.inf, math.inf)

        def hash_xy(x, y):
            return (x << 32) | (y & 0xffffffff)

        for i in range(n):
            x, y = nums1[i], nums2[i]
            h = hash_xy(x, y)
            if h in seen:
                cur = (seen[h], i)
                if cur < res:
                    res = cur
            else:
                seen[h] = i

        if res[0] != math.inf:
            return [res[0], res[1]]

        points = [[nums1[i], nums2[i], i] for i in range(n)]
        points.sort()
        _, res_i, res_j = closest(points)
        if res_i > res_j:
            res_i, res_j = res_j, res_i
        return [res_i, res_j]

def closest(points: List[List[int]]) -> Tuple[int, int, int]:
    n = len(points)
    if n < 2:
        return math.inf, math.inf, math.inf
    mid = n // 2
    midP = points[mid]

    dl, pi, pj = closest(points[:mid])
    dr, qi, qj = closest(points[mid:])
    dres = min((dl, pi, pj), (dr, qi, qj))
    d = dres[0]

    strip = [p for p in points if abs(p[0] - midP[0]) <= d]
    ds, si, sj = strip_closest(strip, d)
    return min(dres, (ds, min(si, sj), max(si, sj)))


def strip_closest(strip: List[List[int]], d: int) -> Tuple[int, int, int]:
    min_d = d
    si = sj = math.inf
    strip.sort(key=lambda a: (a[1], a[2]))
    m = len(strip)
    for i in range(m):
        for j in range(i+1, m):
            if strip[j][1] - strip[i][1] > min_d:
                break
            curr_d = abs(strip[j][1] - strip[i][1]) + abs(strip[j][0] - strip[i][0])
            curr_i = min(strip[i][2], strip[j][2])
            curr_j = max(strip[i][2], strip[j][2])
            if (curr_d, curr_i, curr_j) < (min_d, si, sj):
                min_d, si, sj = curr_d, curr_i, curr_j
    return min_d, si, sj
