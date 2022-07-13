# https://leetcode.com/problems/merge-k-sorted-lists/

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

    def __lt__(self, other):
        return self.val < other.val

import heapq

class Solution(object):
    def mergeKLists(self, lists):
        if not lists:
            return None

        min_heap = []
        for head in lists:
            if head:
                heapq.heappush(min_heap, head)
            
        if not min_heap: # lists contain all None
            return None
        
        head = cur = None
        
        while min_heap:
            min_node = heapq.heappop(min_heap)
            if not head:
                head = min_node
                
            if cur:
                cur.next = min_node
            cur = min_node
            
            next_node = min_node.next
            if next_node:
                heapq.heappush(min_heap, next_node)
            
        return head
        