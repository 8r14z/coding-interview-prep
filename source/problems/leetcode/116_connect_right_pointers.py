# https://leetcode.com/problems/populating-next-right-pointers-in-each-node/

class Solution:
    def connect(self, root: 'Node') -> 'Node':
        if not root:
            return None
        
        head = root
        
        while head:
            
            cur = head
            while cur:
                if cur.left:
                    cur.left.next = cur.right
                    
                cur_child_tail = cur.right or cur.left
                if cur_child_tail and cur.next:
                    next_child_head = cur.next.left or cur.next.right
                    cur_child_tail.next = next_child_head
                
                cur = cur.next
            
            head = head.left or head.right
        
        return root
        