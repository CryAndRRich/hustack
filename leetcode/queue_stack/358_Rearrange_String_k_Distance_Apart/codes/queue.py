from collections import Counter, deque
import heapq

class Solution:
    def rearrangeString(self, s: str, k: int) -> str:
        if k == 0:
            return s

        freq = Counter(s)
        max_heap = [(-count, ch) for ch, count in freq.items()]
        heapq.heapify(max_heap)

        queue = deque()  
        result = []

        while max_heap or queue:
            if queue and queue[0][2] <= len(result):
                count, ch, _ = queue.popleft()
                heapq.heappush(max_heap, (count, ch))

            if not max_heap:
                return "" if len(result) < len(s) else "".join(result)

            count, ch = heapq.heappop(max_heap)
            result.append(ch)

            if -count > 1:
                queue.append((count + 1, ch, len(result) + k - 1))

        return "".join(result)
