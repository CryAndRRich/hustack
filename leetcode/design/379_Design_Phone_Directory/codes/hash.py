class PhoneDirectory:
    def __init__(self, maxNumbers: int):
        self.slotsAvailable = set(range(maxNumbers))

    def get(self) -> int:
        if not self.slotsAvailable:
            return -1
        slot = next(iter(self.slotsAvailable))
        self.slotsAvailable.remove(slot)
        return slot

    def check(self, number: int) -> bool:
        return number in self.slotsAvailable

    def release(self, number: int) -> None:
        self.slotsAvailable.add(number)

# Your PhoneDirectory object will be instantiated and called as such:
# obj = PhoneDirectory(maxNumbers)
# param_1 = obj.get()
# param_2 = obj.check(number)
# obj.release(number)