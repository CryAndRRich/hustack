from collections import defaultdict
from typing import List

class Solution:
    def assignBikes(self, workers: List[List[int]], bikes: List[List[int]]) -> List[int]:
        worker_count = len(workers)
        unassigned_workers = set(range(worker_count))
        bike_count = len(bikes)
        unassigned_bikes = set(range(bike_count))
        res = [-1] * worker_count
        deviation = 1  

        while unassigned_workers:
            lookup = defaultdict(list)
            for bike_id in unassigned_bikes:
                bx, by = bikes[bike_id]
                key = (bx // deviation, by // deviation)
                lookup[key].append(bike_id)

            pool = []
            for worker_id in unassigned_workers:
                wx, wy = workers[worker_id]
                grid_x, grid_y = wx // deviation, wy // deviation

                for dx in range(-1, 2):
                    for dy in range(-1, 2):
                        cell = (grid_x + dx, grid_y + dy)
                        for bike_id in lookup.get(cell, []):
                            bx, by = bikes[bike_id]
                            dist = abs(wx - bx) + abs(wy - by)
                            if dist <= deviation:
                                pool.append((dist, worker_id, bike_id))

            pool.sort()
            for _, worker_id, bike_id in pool:
                if res[worker_id] == -1 and bike_id in unassigned_bikes:
                    res[worker_id] = bike_id
                    unassigned_workers.remove(worker_id)
                    unassigned_bikes.remove(bike_id)

            deviation <<= 1  
        return res
