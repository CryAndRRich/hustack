from collections import defaultdict
import heapq
from typing import List

class Solution:
    def highFive(self, items: List[List[int]]) -> List[List[int]]:
        scores = defaultdict(list)
        for id, score in items:
            heapq.heappush(scores[id], score)
            if len(scores[id]) > 5:
                heapq.heappop(scores[id])
        
        result = []
        for id in sorted(scores):
            avg = sum(scores[id]) // 5
            result.append([id, avg])
        return result
