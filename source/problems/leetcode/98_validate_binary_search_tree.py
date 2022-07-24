# https://leetcode.com/problems/validate-binary-search-tree/
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
INF = float('inf')
class Solution:
    def isValidBST(self, root: Optional[TreeNode], min=-INF, max =INF) -> bool:
        if not root:
            return True
        
        if root.val <= min and root.val >= max:
            return False
        
        return self.isValidBST(root.left, min, root.val) and self.isValidBST(root.right, root.val, max)