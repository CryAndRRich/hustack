# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def plusOne(self, head: ListNode) -> ListNode:
        def dfs(node: ListNode) -> int:
            if not node:
                return 1 
            carry = dfs(node.next)
            node.val += carry
            if node.val == 10:
                node.val = 0
                return 1
            return 0

        carry = dfs(head)
        if carry == 1:
            new_head = ListNode(1)
            new_head.next = head
            return new_head
        return head
