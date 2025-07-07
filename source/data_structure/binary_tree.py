from typing import Optional

class Node:
    def __init__(self, data):
        self.left: Optional[Node] = None
        self.right: Optional[Node] = None
        self.val = data

node1 = Node(1)
node2 = Node(2)
node3 = Node(3)
node4 = Node(4)
node5 = Node(5)

node2.left = node1
node2.right = node4
node4.left = node3
node4.right = node5

def interative_inorder(node) -> list[int]:
    res = []
    stack = []
    cur = node
    
    while stack or cur:
        if cur:
            stack.append(cur)
            cur = cur.left
        else:
            cur = stack.pop()
            res.append(cur.val)
            cur = cur.right
            
    return res 

def recursive_inorder(node) -> list[int]:
    if node is None: return []
    return recursive_inorder(node.left) + [node.val] + recursive_inorder(node.right) 

print(interative_inorder(node2))