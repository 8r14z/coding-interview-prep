import collections

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    node  = 0
    def kthSmallest(self, root: TreeNode, k: int) -> int:
        stack = collections.deque()

        while True:
            while root:
                stack.append(root)
                root =  root.left

            root = stack.pop()
            k -= 1
            if not k:
                return root.val

            root = root.right



        
