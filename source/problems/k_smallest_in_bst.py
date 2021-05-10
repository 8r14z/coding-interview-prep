import collections

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def kthSmallest(self, root: TreeNode, k: int) -> int:
        # Solution 1
        # stack = collections.deque()
        # while True:
        #     while root:
        #         stack.append(root)
        #         root =  root.left

        #     root = stack.pop()
        #     k -= 1
        #     if not k:
        #         return root.val

        #     root = root.right

        # Solution 2
        res = [0, None]
        self.inorder(root, res, k)
        return res[1]
    
    def inorder(self, node: TreeNode, tracker: [], k: int):
        if node is None: return
        
        self.inorder(node.left, tracker, k)

        tracker[0] += 1
        if tracker[0] == k:
            tracker[1] = node.val
            return
            
        self.inorder(node.right, tracker, k)
    
        
# NOTE: Space complexity should be counted for stack calls as well