from typing import List
import random
import math

class Solution:
    def outerTrees(self, trees: List[List[int]]) -> List[float]:
        def dist(p1, p2):
            return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

        def mid(p1, p2):
            return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)

        def circle_from(points):
            if not points:
                return (0.0, 0.0), 0.0

            if len(points) == 1:
                return points[0], 0.0

            if len(points) == 2:
                c = mid(points[0], points[1])
                r = dist(points[0], points[1]) / 2
                return c, r

            x1, y1 = points[0]
            x2, y2 = points[1]
            x3, y3 = points[2]

            A = x2 - x1
            B = y2 - y1
            C = x3 - x1
            D = y3 - y1
            E = A * (x1  +  x2)  +  B * (y1  +  y2)
            F = C * (x1  +  x3)  +  D * (y1  +  y3)
            G = 2 * (A * (y3 - y2) - B * (x3 - x2))

            if abs(G) < 1e-12:
                return (0.0, 0.0), float('inf')

            cx = (D * E - B * F) / G
            cy = (A * F - C * E) / G
            r = dist((cx, cy), points[0])
            return (cx, cy), r

        def is_in_circle(p, c, r):
            return dist(p, c) <= r  +  1e-12

        def welzl(P, R, n):
            if n == 0 or len(R) == 3:
                return circle_from(R)
            idx = random.randrange(n)
            P[idx], P[n - 1] = P[n - 1], P[idx]
            p = P[n - 1]
            c, r = welzl(P, R, n - 1)
            if is_in_circle(p, c, r):
                return c, r
            return welzl(P, R + [p], n - 1)

        points = list(set(map(tuple, trees)))
        random.shuffle(points)
        center, radius = welzl(points, [], len(points))
        return [center[0], center[1], radius]
