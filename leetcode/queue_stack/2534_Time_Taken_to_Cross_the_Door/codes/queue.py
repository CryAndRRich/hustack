from collections import deque
from typing import List

class Solution:
    def timeTaken(self, arrival: List[int], state: List[int]) -> List[int]:
        n = len(arrival)
        res = [-1 for _ in range(n)]
        enter_queue = deque()
        exit_queue = deque()

        time = 0
        i = 0  
        last_used = -1  

        while i < n or enter_queue or exit_queue:
            while i < n and arrival[i] <= time:
                if state[i] == 0:
                    enter_queue.append(i)
                else:
                    exit_queue.append(i)
                i += 1

            if exit_queue and (last_used == 1 or last_used == -1 or not enter_queue):
                person = exit_queue.popleft()
                res[person] = time
                last_used = 1
            elif enter_queue:
                person = enter_queue.popleft()
                res[person] = time
                last_used = 0
            else:
                last_used = -1  

            time += 1

        return res
