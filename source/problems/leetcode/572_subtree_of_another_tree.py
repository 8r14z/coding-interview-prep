# https://leetcode.com/problems/subtree-of-another-tree/

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def isSameTree(self, root1, root2) -> bool:
        if not root1 and not root2:
            return True
        elif not root1 and root2:
            return False
        elif root1 and not root2:
            return False

        return (
            root1.val == root2.val 
            and self.isSameTree(root1.left, root2.left) 
            and self.isSameTree(root1.right, root2.right)
        )
    
    def isSubtree(self, root: TreeNode, subRoot: TreeNode) -> bool:
        if not subRoot: 
            return True
        if not root: 
            return False
        
        return self.isSameTree(root, subRoot) or self.isSubtree(root.left, subRoot) or self.isSubtree(root.right, subRoot)