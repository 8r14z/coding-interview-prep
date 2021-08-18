# https://leetcode.com/problems/count-good-nodes-in-binary-tree/

# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def goodNodes(self, root: TreeNode, max_val = float('-inf')) -> int:
        if not root:
            return 0
        
        ans = 0
        if root.val >= max_val:
            ans += 1
            max_val = root.val
        
        if root.left:
            ans += self.goodNodes(root.left, max_val)
        if root.right:
            ans += self.goodNodes(root.right, max_val)
            
        return ans
        