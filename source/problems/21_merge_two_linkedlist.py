# https://leetcode.com/problems/merge-two-sorted-lists/
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        if l1 != None and l2 is None:
            return l1
        elif l2 != None and l1 is None:
            return l2
        elif l1 is None and l2 is None:
            return None
        
        head = None
        if l1.val < l2.val: 
            head = l1
            l1 = l1.next
        else:
            head = l2
            l2 = l2.next
        head.next = self.mergeTwoLists(l1, l2)

        return head