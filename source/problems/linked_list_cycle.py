# https://leetcode.com/problems/linked-list-cycle/submissions/

# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def hasCycle(self, head: ListNode) -> bool:
        if head is None: return False
        
        slow = head
        fast = head
        
        while True:
            nextSlow = slow.next
            nextFast = fast.next
            if nextFast is not None:
                nextFast = nextFast.next
                
            if nextSlow is None or nextFast is None:
                break
            
            if nextSlow == nextFast:
                return True
            
            slow = nextSlow
            fast = nextFast
    
        return False
        
        