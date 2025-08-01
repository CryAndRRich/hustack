from collections import deque, defaultdict
from typing import List

class Solution:
    def minimumSemesters(self, n: int, relations: List[List[int]]) -> int:
        graph = defaultdict(list)
        in_degree = [0] * (n + 1)

        for u, v in relations:
            graph[u].append(v)
            in_degree[v] += 1

        queue = deque()
        for i in range(1, n + 1):
            if in_degree[i] == 0:
                queue.append(i)

        semesters = 0
        taken = 0

        while queue:
            for _ in range(len(queue)):
                node = queue.popleft()
                taken += 1
                for nei in graph[node]:
                    in_degree[nei] -= 1
                    if in_degree[nei] == 0:
                        queue.append(nei)
            semesters += 1

        return semesters if taken == n else -1
