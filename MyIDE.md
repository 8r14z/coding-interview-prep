class Solution:
    def detectCycle(self, head: ListNode) -> ListNode:
        if head is None: return None

        reached = set()
        while head is not None:
            reached.add(head)
            if head.next in reached:
                return head.next
            head = head.next
        
        return None
        