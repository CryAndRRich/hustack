from typing import List

class Solution:
    def isConvex(self, points: List[List[int]]) -> bool:
        def cross_product(A, B, C):
            return (B[0] - A[0]) * (C[1] - B[1]) - (B[1] - A[1]) * (C[0] - B[0])

        n = len(points)
        direction = 0  

        for i in range(n):
            A = points[i]
            B = points[(i + 1) % n]
            C = points[(i + 2) % n]
            cross = cross_product(A, B, C)

            if cross != 0:
                if direction == 0:
                    direction = 1 if cross > 0 else -1
                elif (cross > 0 and direction < 0) or (cross < 0 and direction > 0):
                    return False
        return True
