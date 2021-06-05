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

node3.left = node1
node3.right = node4
node4.left = node2