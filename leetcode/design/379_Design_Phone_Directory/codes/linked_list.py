class ListNode:
    def __init__(self, val=0):
        self.val = val
        self.next = None

class PhoneDirectory:
    def __init__(self, maxNumbers: int):
        self.isSlotAvailable = [True] * maxNumbers
        self.head = ListNode(-1)
        cur = self.head
        for i in range(maxNumbers):
            cur.next = ListNode(i)
            cur = cur.next

    def get(self) -> int:
        if not self.head.next:
            return -1
        node = self.head.next
        self.head.next = node.next
        self.isSlotAvailable[node.val] = False
        return node.val

    def check(self, number: int) -> bool:
        return self.isSlotAvailable[number]

    def release(self, number: int) -> None:
        if self.isSlotAvailable[number]:
            return
        node = ListNode(number)
        node.next = self.head.next
        self.head.next = node
        self.isSlotAvailable[number] = True

# Your PhoneDirectory object will be instantiated and called as such:
# obj = PhoneDirectory(maxNumbers)
# param_1 = obj.get()
# param_2 = obj.check(number)
# obj.release(number)