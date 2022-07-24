# https://leetcode.com/problems/validate-binary-search-tree/

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
INF = float('inf')
class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        def is_valid(node, min, max) -> bool:
            if not node:
                return True
            
            if node.val <= min or node.val >= max:
                return False
            
            return is_valid(node.left, min, node.val) and is_valid(node.right, node.val, max)
            
        return is_valid(root, -INF, INF)