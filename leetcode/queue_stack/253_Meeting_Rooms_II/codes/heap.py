from typing import List
import heapq

class Solution:
    def minMeetingRooms(self, intervals: List[List[int]]) -> int:
        if not intervals:
            return 0

        intervals.sort(key=lambda x: x[0])  
        heap = []  

        for meeting in intervals:
            if heap and meeting[0] >= heap[0]:
                heapq.heappop(heap) 
            heapq.heappush(heap, meeting[1]) 
        return len(heap)
