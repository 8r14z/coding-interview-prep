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
        def getLevel(node):
            level = 0
            while node:
                level += 1
                node = node.parent
            return level

        temp = p
        pLevel = getLevel(p)
        qLevel = getLevel(q)

        if pLevel > qLevel:
            for _ in range(pLevel - qLevel):
                p = p.parent
        if qLevel > pLevel:
            for _ in range(qLevel - pLevel):
                q = q.parent
        
        while q != p:
            p = p.parent
            q = q.parent
        
        return q

# https://leetcode.com/problems/dot-product-of-two-sparse-vectors/
# simple yet needing to skip zero values to efficiently store :) 
class SparseVector:
    def __init__(self, nums: List[int]):
        self.numPerIndicies = {} 
        for i, num in enumerate(nums):
            if num != 0:
                self.numPerIndicies[i] = num

    # Return the dotProduct of two sparse vectors
    def dotProduct(self, vec: 'SparseVector') -> int:
        res = 0
        for i, num in vec.numPerIndicies.items():
            if i in self.numPerIndicies:
                thisNum = self.numPerIndicies[i]
                res += (num * thisNum)
        return res

# https://leetcode.com/problems/valid-palindrome-ii/
def isPalindrome(s, start, end):
    while start < end:
        if s[start] != s[end]:
            return False
        start += 1
        end -= 1

    return True

class Solution:
    def validPalindrome(self, s: str) -> bool:
        left = 0
        right = len(s)-1
        
        while left < right:
            if s[left] == s[right]:
                left += 1
                right -=1
            else:
                return isPalindrome(s, left+1, right) or isPalindrome(s, left, right-1)

        return True

