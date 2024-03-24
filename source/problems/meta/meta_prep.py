# https://leetcode.com/problems/binary-tree-vertical-order-traversal/description/
# Time: O(N) -- Space: O(N)
from collections import deque
from collections import defaultdict
class Solution:
    def verticalOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []

        queue = deque([(root, 0)])
        colMap = defaultdict(list)

        minCol = float('inf')
        maxCol = float('-inf')

        while queue:
            node, col = queue.popleft()
            minCol = min(minCol, col)
            maxCol = max(maxCol, col)
            
            colMap[col].append(node.val)

            if node.left:
                queue.append((node.left, col-1))
            if node.right:
                queue.append((node.right, col+1))
        
        res = []
        for col in range(minCol, maxCol+1):
            res.append(colMap[col])

        return res
                
# https://leetcode.com/problems/valid-word-abbreviation/
class Solution:
    def validWordAbbreviation(self, word: str, abbr: str) -> bool:
        skipCount = 0
        i = 0

        for c in abbr:
            if c.isdigit():
                if skipCount == 0 and c == '0': #leading zero
                    return False
                skipCount = skipCount * 10 + int(c)
            else:
                i += skipCount
                skipCount = 0
                
                if i >= len(word) or word[i] != c:
                    return False

                i+=1

        i += skipCount

        return i == len(word)