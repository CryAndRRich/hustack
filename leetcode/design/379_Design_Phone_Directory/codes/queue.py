from collections import deque

class PhoneDirectory:
    def __init__(self, maxNumbers: int):
        self.isSlotAvailable = [True] * maxNumbers
        self.slotsAvailableQueue = deque(range(maxNumbers))

    def get(self) -> int:
        if not self.slotsAvailableQueue:
            return -1
        slot = self.slotsAvailableQueue.popleft()
        self.isSlotAvailable[slot] = False
        return slot

    def check(self, number: int) -> bool:
        return self.isSlotAvailable[number]

    def release(self, number: int) -> None:
        if self.isSlotAvailable[number]:
            return
        self.slotsAvailableQueue.append(number)
        self.isSlotAvailable[number] = True

# Your PhoneDirectory object will be instantiated and called as such:
# obj = PhoneDirectory(maxNumbers)
# param_1 = obj.get()
# param_2 = obj.check(number)
# obj.release(number)