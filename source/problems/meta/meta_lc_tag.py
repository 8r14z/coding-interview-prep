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

class Solution
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

# https://leetcode.com/problems/random-pick-with-weight
class Solution:

    def __init__(self, w: List[int]):
        prefix_sums = [w[0]]
        for i in range(1, len(w)):
            prefix_sums.append(w[i] + prefix_sums[i-1])
        
        self.prefix_sums = prefix_sums
        self.total = prefix_sums[-1]

    def pickIndex(self) -> int:
        target = self.total * random.random()
        low = 0 
        high = len(self.prefix_sums)
        result = high
        while low <= high:
            mid = (low+high)//2
            if self.prefix_sums[mid] > target:
                result = mid
                high = mid - 1
            else:
                low = mid + 1
        return result

# https://leetcode.com/problems/basic-calculator-ii/
class Solution:
    def calculate(self, s: str) -> int:
        numStack = []
        op = '+'
        curNum = ''

        for i,c in enumerate(s):
            if c.isdigit():
                curNum += c
            
            if i == len(s) - 1 or c in ['+', '-', '*', '/']:
                integer = int(curNum)
                
                if op == '+':
                    pass
                elif op == '-':
                    integer *= -1
                elif op == '*':
                    if numStack:
                        lastNum = numStack.pop()
                        integer = lastNum * integer
                else:
                    if numStack:
                        lastNum = numStack.pop()
                        integer = lastNum // integer if lastNum >= 0 else -(abs(lastNum) // integer) # -3 / 2 == 2 in python, all devision is rounded down.

                numStack.append(integer)

                op = c
                curNum = ''

        res = 0
        for num in numStack:
            res += num
        return res

# https://leetcode.com/problems/buildings-with-an-ocean-view/
class Solution:
    def findBuildings(self, heights: List[int]) -> List[int]:
        if not heights:
            return []

        res = []
        maxHeight = -1
        for i in range(len(heights)-1, -1, -1):
            curHeight = heights[i]
            if curHeight > maxHeight:
                res.append(i)
                maxHeight = curHeight
                
        res.reverse()
        return res

# https://leetcode.com/problems/kth-largest-element-in-an-array/
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        n = len(nums)
        if k > n: return -1
        
        def partition(left, right, arr) -> int:
            pivot = arr[right]
            pivot_index = left # save last index that > pivot
            
            for i in range(left,right):
                if arr[i] <= pivot:
                    arr[i], arr[pivot_index] = arr[pivot_index], arr[i]
                    pivot_index += 1
                    
            arr[right], arr[pivot_index] = arr[pivot_index], arr[right]
            
            return pivot_index
            
        left = 0
        right = n-1
        target = n-k
        
        while left <= right:
            pivot_index = partition(left, right, nums)
            if pivot_index == target:
                return nums[pivot_index]
            elif pivot_index > target:
                right = pivot_index - 1
            else:
                left = pivot_index + 1
                
        return -1
    
# https://leetcode.com/problems/range-sum-of-bst/
class Solution:
    def rangeSumBST(self, root: Optional[TreeNode], low: int, high: int) -> int:
        if not root:
            return 0

        sum = 0
        if root.val >= low and root.val <= high:
            sum += root.val
        
        if root.val > low: # left child < root.val
            sum += self.rangeSumBST(root.left, low, high)

        if root.val < high: # right child > root.val
            sum += self.rangeSumBST(root.right, low, high)
        
        return sum

# https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/
# DFS down to leaf and stop when find q or p is found
# if we can find both q and p on both left and right of a given node, the node is lowest common ancestor
# it we can only either q or p, so q or p is the lowest common ancestor. 
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if not root:
            return None
        
        if root == p or root == q:
            return root

        # find p/q on the left child
        leftAncestor = self.lowestCommonAncestor(root.left, p, q) 
        # find p/q on the right child
        rightAncestor = self.lowestCommonAncestor(root.right, p, q)

        if leftAncestor and rightAncestor:
            return root
        else:
            return leftAncestor or rightAncestor

# https://leetcode.com/problems/convert-binary-search-tree-to-sorted-doubly-linked-list/
class Solution:
    def treeToDoublyList(self, root: 'Optional[Node]') -> 'Optional[Node]':
        if not root:
            return None
        
        def inorder(node):
            if not node:
                return

            inorder(node.left)

            if not self.head:
                # save the head once
                self.head = node
            
            if self.tail:
                node.left = self.tail
                self.tail.right = node

            self.tail = node

            inorder(node.right)

        self.head = self.tail = None
        inorder(root)
        self.head.left = self.tail
        self.tail.right = self.head
        
        return self.head

# https://leetcode.com/problems/simplify-path/
class Solution:
    def simplifyPath(self, path: str) -> str:
        pathStack = []

        for subpath in path.split('/'):
            if subpath == '.' or len(subpath) == 0:
                continue
            elif subpath == '..':
                if pathStack:
                    pathStack.pop()
            else:
                pathStack.append(subpath)
        
        return '/' + '/'.join(pathStack)