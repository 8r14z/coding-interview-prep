# https://leetcode.com/problems/binary-tree-maximum-path-sum/
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
INF = float('inf')
class Solution:
    def maxPathSum(self, root: TreeNode) -> int:
        def dfs(node) -> int:
            if not node: 
                return 0
            
            cur_val = node.val
            left_val = dfs(node.left)
            right_val = dfs(node.right)
            
            cur_max = max(cur_val,
                          cur_val + left_val, 
                          cur_val + right_val, 
                          cur_val + left_val + right_val)
            
            self.global_max = max(self.global_max, cur_max)
            
            return max(cur_val, cur_val + left_val, cur_val + right_val)
            # sequence means the path should be in same branch == A node can only appear in the sequence at most once. Note that the path does not need to pass through the root.
        
        self.global_max = -INF
        dfs(root)
        return self.global_max
        
        
        