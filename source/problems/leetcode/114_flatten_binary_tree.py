# https://leetcode.com/problems/flatten-binary-tree-to-linked-list/

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class Solution:
    def flatten(self, root: TreeNode) -> None:
        def preorder(node):
            if not node: return
            
            left = node.left
            right = node.right
            
            if self.prev:
                self.prev.right = node
                self.prev.left = None
            self.prev = node
            
            preorder(left)
            preorder(right)
            
        self.prev = None
        preorder(root)
        return root
        