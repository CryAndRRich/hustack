from collections import deque

class HitCounter:
    def __init__(self):
        self.total = 0
        self.hits = deque()

    def hit(self, timestamp: int) -> None:
        if not self.hits or self.hits[-1][0] != timestamp:
            self.hits.append([timestamp, 1])
        else:
            self.hits[-1][1] += 1
        self.total += 1

    def getHits(self, timestamp: int) -> int:
        while self.hits and timestamp - self.hits[0][0] >= 300:
            self.total -= self.hits[0][1]
            self.hits.popleft()
        return self.total
