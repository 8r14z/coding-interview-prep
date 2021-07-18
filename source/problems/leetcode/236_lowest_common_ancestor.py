# https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/

class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if not root:
            return None
        
        cur_node = root
        
        if cur_node in [p,q]:
            return cur_node
        
        left_result = self.lowestCommonAncestor(cur_node.left, p, q)
        right_result = self.lowestCommonAncestor(cur_node.right, p, q)
        
        if left_result and right_result:
            return cur_node
        else:
            return left_result or right_result
            
