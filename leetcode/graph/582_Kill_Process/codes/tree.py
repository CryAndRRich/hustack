from collections import defaultdict, deque
from typing import List

class Solution:
    def killProcess(self, pid: List[int], ppid: List[int], kill: int) -> List[int]:
        tree = defaultdict(list)
        for i in range(len(pid)):
            tree[ppid[i]].append(pid[i])

        result = []
        queue = deque([kill])

        while queue:
            curr = queue.popleft()
            result.append(curr)
            for child in tree[curr]:
                queue.append(child)
        return result
