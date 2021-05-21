# https://leetcode.com/problems/linked-list-cycle-ii/
# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def intersection(self, head: ListNode) -> ListNode:
        slow = head
        fast = head
        
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if fast is slow:
                return fast
        
        return None
    
    def detectCycle(self, head: ListNode) -> ListNode:
        if head is None or head.next is None: return None
        
        intersect = self.intersection(head)
        if intersect is None:
            return None
            
        start = head
        while start is not intersect:
            start = start.next
            intersect = intersect.next
        return start
        
        
# Analysis https://bohenan.gitbooks.io/leetcode/content/linkedlist/linked-list-cycle-ii.html

