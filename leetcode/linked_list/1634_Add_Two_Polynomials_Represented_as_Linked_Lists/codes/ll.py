# Definition for polynomial singly-linked list
class PolyNode:
    def __init__(self, x=0, y=0, next=None):
        self.coefficient = x
        self.power = y
        self.next = next

class Solution:
    def addPoly(self, poly1: 'PolyNode', poly2: 'PolyNode') -> 'PolyNode':
        dummy = PolyNode()
        curr = dummy
        while poly1 and poly2:
            if poly1.power == poly2.power:
                coef = poly1.coefficient + poly2.coefficient
                if coef != 0:
                    curr.next = PolyNode(coef, poly1.power)
                    curr = curr.next
                poly1 = poly1.next
                poly2 = poly2.next
            elif poly1.power > poly2.power:
                curr.next = PolyNode(poly1.coefficient, poly1.power)
                curr = curr.next
                poly1 = poly1.next
            else:
                curr.next = PolyNode(poly2.coefficient, poly2.power)
                curr = curr.next
                poly2 = poly2.next
        while poly1:
            curr.next = PolyNode(poly1.coefficient, poly1.power)
            curr = curr.next
            poly1 = poly1.next
        while poly2:
            curr.next = PolyNode(poly2.coefficient, poly2.power)
            curr = curr.next
            poly2 = poly2.next
        return dummy.next
