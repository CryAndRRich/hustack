from typing import List

class Solution:
    def assignBikes(self, workers: List[List[int]], bikes: List[List[int]]) -> List[int]:
        pairs = []
        for i, (wx, wy) in enumerate(workers):
            for j, (bx, by) in enumerate(bikes):
                dist = abs(wx - bx) + abs(wy - by)
                pairs.append((dist, i, j))

        pairs.sort()
        assigned = [-1] * len(workers)
        used_bikes = set()

        for _, i, j in pairs:
            if assigned[i] == -1 and j not in used_bikes:
                assigned[i] = j
                used_bikes.add(j)

        return assigned
