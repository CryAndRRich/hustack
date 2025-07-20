from collections import defaultdict, deque
from typing import List

class Solution:
    def alienOrder(self, words: List[str]) -> str:
        graph = defaultdict(set)
        in_degree = {c: 0 for word in words for c in word}

        for i in range(len(words) - 1):
            w1, w2 = words[i], words[i + 1]
            if len(w1) > len(w2) and w1.startswith(w2):
                return ""
            for c1, c2 in zip(w1, w2):
                if c1 != c2:
                    if c2 not in graph[c1]:
                        graph[c1].add(c2)
                        in_degree[c2] += 1
                    break

        queue = deque([c for c in in_degree if in_degree[c] == 0])
        res = []

        while queue:
            c = queue.popleft()
            res.append(c)
            for nei in graph[c]:
                in_degree[nei] -= 1
                if in_degree[nei] == 0:
                    queue.append(nei)

        return "".join(res) if len(res) == len(in_degree) else ""
