# LC Tag: https://leetcode.com/company/facebook/

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
    
# https://leetcode.com/problems/minimum-remove-to-make-valid-parentheses/
class Solution:
    def minRemoveToMakeValid(self, s: str) -> str:
        stack = []
        invalidClose = set()
        for i in range(len(s)):
            c = s[i]
            if c == '(':
                stack.append(c)
            elif c == ')':
                if not stack:
                    invalidClose.add(i)
                else:
                    stack.pop()
                    
        stack = []
        invalidOpen = set()
        for i in range(len(s)-1, -1, -1):
            c = s[i]
            if c == ')':
                stack.append(c)
            elif c == '(':
                if not stack:
                    invalidOpen.add(i)
                else:
                    stack.pop()

        res = ''
        for i, c in enumerate(s):
            if i not in invalidOpen and i not in invalidClose:
                res += c

        return res
    
# https://leetcode.com/problems/nested-list-weight-sum/ 
class Solution:
    def depthSum(self, nestedList: List[NestedInteger]) -> int:
        def sumAtDepth(sublist, depth):
            res = 0
            for element in sublist:
                if element.isInteger():
                    res += (element.getInteger() * depth)
                else:
                    res += sumAtDepth(element.getList(), depth+1)
            return res

        return sumAtDepth(nestedList, 1)
    
# https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree-iii/
class Solution:
    def lowestCommonAncestor(self, p: 'Node', q: 'Node') -> 'Node':
        temp = p
        levelP = 0
        while temp:
            levelP += 1
            temp = temp.parent
        temp = q
        levelQ = 0
        while temp:
            levelQ += 1
            temp = temp.parent

        if levelP > levelQ:
            for _ in range(levelP - levelQ):
                p = p.parent
        if levelQ > levelP:
            for _ in range(levelQ - levelP):
                q = q.parent
        
        while q != p:
            p = p.parent
            q = q.parent
        
        return q
