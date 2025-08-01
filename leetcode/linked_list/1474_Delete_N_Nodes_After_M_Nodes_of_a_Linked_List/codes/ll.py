from typing import Optional

# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def deleteNodes(self, head: Optional[ListNode], m: int, n: int) -> Optional[ListNode]:
        curr = head
        while curr:
            for _ in range(1, m):
                if curr:
                    curr = curr.next
            if not curr:
                break

            temp = curr.next
            for _ in range(n):
                if temp:
                    temp = temp.next
                else:
                    break
            curr.next = temp
            curr = temp
        return head 