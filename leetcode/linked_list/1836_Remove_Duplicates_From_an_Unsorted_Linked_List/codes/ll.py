from collections import Counter

# Definition for singly-linked list
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def deleteDuplicatesUnsorted(self, head: ListNode) -> ListNode:
        vals = []
        curr = head
        while curr:
            vals.append(curr.val)
            curr = curr.next
        count = Counter(vals)
        dummy = ListNode(0)
        dummy.next = head
        prev = dummy
        curr = head
        while curr:
            if count[curr.val] > 1:
                prev.next = curr.next
            else:
                prev = curr
            curr = curr.next
        return dummy.next
