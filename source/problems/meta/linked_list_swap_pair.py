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
        # prev -> first -> second -> second.next
        # to
        # prev -> second -> first -> second.next
        first.next = second.next
        second.next = first 
        if prev:
            prev.next = second
        
        prev = first 
        cur = first.next 