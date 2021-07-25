# https://leetcode.com/problems/rotate-list/
# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    def rotateRight(self, head: ListNode, k: int) -> ListNode:
        if not head:
            return None
        tail = head
        n = 1
        while tail.next:
            n += 1
            tail = tail.next
        k = k % n
        tail.next = head
        
        tail = head
        for _ in range(n-k-1):
            tail = tail.next
        new_head = tail.next
        tail.next = None
        return new_head