class ZigzagIterator:
    def __init__(self, v1: list[int], v2: list[int]):
        self.data = [v1, v2]
        self.ptrs = [0, 0]
        self.turn = 0

    def next(self) -> int:
        for _ in range(2):
            i = self.turn % 2
            if self.ptrs[i] < len(self.data[i]):
                res = self.data[i][self.ptrs[i]]
                self.ptrs[i] += 1
                self.turn += 1
                return res
            self.turn += 1
        return -1

    def hasNext(self) -> bool:
        return any(self.ptrs[i] < len(self.data[i]) for i in [0, 1])


# Your ZigzagIterator object will be instantiated and called as such:
# i, v = ZigzagIterator(v1, v2), []
# while i.hasNext(): v.append(i.next())