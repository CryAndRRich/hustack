from collections import deque, Counter
from sortedcontainers import SortedList, SortedDict

class StatisticTracker:
    def add(self, elem):
        return NotImplementedError("Subclasses should implement this method")
    def remove(self, elem): 
        return NotImplementedError("Subclasses should implement this method")
    def calc(self):
        return NotImplementedError("Subclasses should implement this method")

class MeanStatisticTracker(StatisticTracker):
    def __init__(self):
        self._sum = 0
        self._cnt = 0

    def add(self, elem):
        self._sum += elem
        self._cnt += 1

    def remove(self, elem):
        self._sum -= elem
        self._cnt -= 1

    def calc(self):
        return self._sum // self._cnt

class MedianStatisticTracker(StatisticTracker):
    def __init__(self):
        self._list = SortedList()
    
    def add(self, elem):
        self._list.add(elem)
    
    def remove(self, elem):
        self._list.remove(elem)
    
    def calc(self):
        idx = len(self._list) // 2
        return self._list[idx]

class ModeStatisticTracker(StatisticTracker):
    def __init__(self):
        self._number_to_cnt = Counter()
        self._cnt_to_numbers = SortedDict()

    def add(self, elem):
        old = self._number_to_cnt[elem]
        self._number_to_cnt[elem] += 1
        if old > 0:
            self._cnt_to_numbers[old].remove(elem)
            if not self._cnt_to_numbers[old]:
                del self._cnt_to_numbers[old]
        self._cnt_to_numbers.setdefault(old + 1, SortedList()).add(elem)

    def remove(self, elem):
        old = self._number_to_cnt[elem]
        if old == 1:
            del self._number_to_cnt[elem]
        else:
            self._number_to_cnt[elem] -= 1
        self._cnt_to_numbers[old].remove(elem)
        if not self._cnt_to_numbers[old]:
            del self._cnt_to_numbers[old]
        if old > 1:
            self._cnt_to_numbers.setdefault(old - 1, SortedList()).add(elem)

    def calc(self):
        return self._cnt_to_numbers.peekitem(-1)[1][0]

class StatisticsTracker:
    def __init__(self):
        self._q = deque()
        self._mean = MeanStatisticTracker()
        self._median = MedianStatisticTracker()
        self._mode = ModeStatisticTracker()
        self._trackers = [self._mean, self._median, self._mode]

    def addNumber(self, number: int) -> None:
        self._q.append(number)
        for tracker in self._trackers:
            tracker.add(number)

    def removeFirstAddedNumber(self) -> None:
        number = self._q.popleft()
        for tracker in self._trackers:
            tracker.remove(number)

    def getMean(self) -> int:
        return self._mean.calc()

    def getMedian(self) -> int:
        return self._median.calc()

    def getMode(self) -> int:
        return self._mode.calc()

# Your StatisticsTracker object will be instantiated and called as such:
# obj = StatisticsTracker()
# obj.addNumber(number)
# obj.removeFirstAddedNumber()
# param_3 = obj.getMean()
# param_4 = obj.getMedian()
# param_5 = obj.getMode()