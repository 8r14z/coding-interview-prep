# https://leetcode.com/problems/count-good-nodes-in-binary-tree/

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def goodNodes(self, root: TreeNode) -> int:
        def dfs(node, val):
            if not node:
                return 0
            
            ans = 0
            if node.val >= val:
                ans = 1
            
            if node.left:
                ans += dfs(node.left, max(node.val, val))
            
            if node.right:
                ans += dfs(node.right, max(node.val, val))
            
            return ans
        
        return dfs(root, float('-inf'))
        