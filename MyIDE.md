# This is the IDE I mainly use to practice

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

def solve(node) -> [int]:
    if node is None: return []
    return solve(node.left) + [node.val] + solve(node.right) 

def solve2(node) -> [int]:
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

print(solve2(node2))

def is_valid(a):
    min_val = -1
    stack = []

    for i in range(len(a)):
        if a[i] < min_val:
            return False
        else:
            while len(stack) > 0 and stack[-1] < a[i]:
                min_val = stack.pop()
            stack.append(a[i])

    return True

print(is_valid([3,1,2,4,2,5]))
print(is_valid([5,1,4,3,6]))