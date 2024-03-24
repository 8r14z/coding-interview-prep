# https://leetcode.com/problems/sum-root-to-leaf-numbers/

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

from collections import defaultdict

class Solution:
    def sumNumbers(self, root: Optional[TreeNode]) -> int:
        res = 0
        if not root:
            return 0
        
        sumAtNode = defaultdict(int)
        sumAtNode[root] = root.val
        stack = [root]
        
        while stack:
            node = stack.pop()

            if not node.left and not node.right:
                res += sumAtNode[node]
            
            if node.left:
                leftNode = node.left
                sumAtNode[leftNode] = sumAtNode[node] * 10 + leftNode.val
                stack.append(leftNode)
            if node.right:
                rightNode = node.right
                sumAtNode[rightNode] = sumAtNode[node] * 10 + rightNode.val
                stack.append(rightNode)

        return res