from typing import List
from collections import defaultdict

class Solution:
    def maxArea(self, height: int, positions: List[int], directions: str) -> int:
        n = len(positions)
        total_pos = sum(positions)
        velocity = 0
        events = defaultdict(list)
        
        for i in range(n):
            if directions[i] == 'U':
                velocity += 1
                events[height - positions[i]].append(i)
                events[height - positions[i] + height].append(i)
            else:  
                velocity -= 1
                events[positions[i]].append(i)
                events[positions[i] + height].append(i)
        
        best = total_pos
        current_time = 0
        dirs = list(directions)
        
        for t in sorted(events.keys()):
            dt = t - current_time
            total_pos += velocity * dt
            current_time = t
            best = max(best, total_pos)
            
            for i in events[t]:
                if dirs[i] == 'U':
                    dirs[i] = 'D'
                    velocity -= 2
                else:
                    dirs[i] = 'U'
                    velocity += 2
        
        return best
