class Node:
    def __init__(self, data):
        self.left = None
        self.right = None
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

def inorder(node) -> [int]:
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

print(inorder(node2))