class Node:
    def __init__(self, val: int):
        self.val = val
        self.next = None

def swapPair(root: Node):
    cur = root
    prev = None 

    while cur:
        first = cur
        second = cur.next 

        # swap
        first.next = second.next
        second.next = first 

        if prev:
            prev.next = second
        
        prev = first 
        cur = first.next 