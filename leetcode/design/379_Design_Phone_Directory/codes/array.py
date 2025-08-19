class PhoneDirectory:
    def __init__(self, maxNumbers: int):
        self.isSlotAvailable = [True] * maxNumbers

    def get(self) -> int:
        for i in range(len(self.isSlotAvailable)):
            if self.isSlotAvailable[i]:
                self.isSlotAvailable[i] = False
                return i
        return -1

    def check(self, number: int) -> bool:
        return self.isSlotAvailable[number]

    def release(self, number: int) -> None:
        self.isSlotAvailable[number] = True

# Your PhoneDirectory object will be instantiated and called as such:
# obj = PhoneDirectory(maxNumbers)
# param_1 = obj.get()
# param_2 = obj.check(number)
# obj.release(number)