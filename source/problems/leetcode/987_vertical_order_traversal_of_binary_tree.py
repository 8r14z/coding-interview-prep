# https://leetcode.com/problems/vertical-order-traversal-of-a-binary-tree/description/

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

from collections import deque
from collections import defaultdict 

class Solution:
    def verticalTraversal(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []

        colMap = defaultdict(list)
        
        queue = deque()
        queue.append((root,0,0))

        minCol = float('inf')
        maxCol = float('-inf')

        while queue:
            node, row, col = queue.popleft()
            minCol = min(minCol, col)
            maxCol = max(maxCol, col)

            colMap[col].append((node.val, row))
            
            if node.left:
                queue.append((node.left, row+1, col-1))
            if node.right:
                queue.append((node.right, row+1, col+1))
        
        res = []

        for col in range(minCol, maxCol + 1):
            if col not in colMap:
                continue
            items = sorted(colMap[col], key=lambda x: (x[1], x[0]))
            vals = [val for val, _ in items]
            res.append(vals)
            
        return res

            
            
            