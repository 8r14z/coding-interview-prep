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

# https://leetcode.com/problems/minimum-add-to-make-parentheses-valid/
class Solution:
    def minAddToMakeValid(self, s: str) -> int:
        stack = []
        res = 0
        for c in s:
            if c == '(':
                stack.append(c)
            else:
                if stack:
                    stack.pop()
                else:
                    res += 1

        return res + len(stack)

# https://leetcode.com/problems/moving-average-from-data-stream/
from collections import deque
class MovingAverage:

    def __init__(self, size: int):
        self.size = size
        self.queue = deque()
        self.total = 0

    def next(self, val: int) -> float:
        self.total += val
        self.queue.append(val)
        if len(self.queue) > self.size:
            headVal = self.queue.popleft()
            self.total -= headVal

        return self.total / len(self.queue)

# https://leetcode.com/problems/powx-n/
# intuiation: x^n = (x^2)^(n/2) = (x^4)^(n/4) = ...(x^logn)^1
class Solution:
    def myPow(self, x: float, n: int) -> float:
        if n == 0:
            return 1
        
        if n < 0:
            x = 1/x
            n *= -1
        
        res = 1
        while n != 0:
            if n % 2:
                res *= x
                n -= 1
            
            x *= x
            n //= 2

        return res

#https://leetcode.com/problems/subarray-sum-equals-k/
from collections import defaultdict
class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        prevSum = defaultdict(int)
        cumulativeSum = 0
        count = 0
        for num in nums:
            cumulativeSum += num
            if cumulativeSum == k:
                count += 1
            if cumulativeSum - k in prevSum:
                count += prevSum[cumulativeSum - k]
            
            prevSum[cumulativeSum] += 1
        
        return count

# https://leetcode.com/problems/custom-sort-string/description/
class Solution:
    def customSortString(self, order: str, s: str) -> str:
        frequency = {}
        for c in order:
            frequency[c] = 0

        for c in s:
            if c in frequency:
                frequency[c] += 1
        
        res = ''
        i = 0
        for c in s:
            if c not in frequency:
                res += c
            else:
                oc = order[i]
                if frequency[oc] > 0:
                    res += oc
                    frequency[oc] -= 1
                    if frequency[oc] == 0:
                        i += 1
        return res

# https://leetcode.com/problems/find-peak-element/        
class Solution:
    def findPeakElement(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 1:
            return 0
        if nums[0] > nums[1]:
            return 0
        if nums[n-1] > nums[n-2]:
            return n-1

        low = 1
        high = n-2
        while low < high:
            mid = (low + high) // 2
            if nums[mid] < nums[mid+1]:
                low = mid + 1
            elif nums[mid] < nums[mid-1]:
                high = mid - 1
            else:
                return mid

        return low
            
# https://leetcode.com/problems/sum-root-to-leaf-numbers/description/
class Solution:
    def sumNumbers(self, root):
        res = 0
        stack = [root]
        sumAtNode = {root: root.val}

        while stack:
            node = stack.pop()
            curSum = sumAtNode[node]

            if not node.left and not node.right:
                res += curSum
            
            if node.left:
                stack.append(node.left)
                sumAtNode[node.left] = curSum * 10 + node.left.val
            if node.right:
                stack.append(node.right)
                sumAtNode[node.right] = curSum * 10 + node.right.val

        return res

# https://leetcode.com/problems/interval-list-intersections/
class Solution:
    def intervalIntersection(self, firstList: List[List[int]], secondList: List[List[int]]) -> List[List[int]]:
        res = []
        i = j = 0
        while i < len(firstList) and j < len(secondList):
            interval1 = firstList[i]
            interval2 = secondList[j]
            start = max(interval1[0], interval2[0])
            end = min(interval1[1], interval2[1])
            if start <= end:
                res.append([start,end])
            if interval1[1] < interval2[1]:
                i += 1
            else:
                j += 1

        return res

# https://leetcode.com/problems/binary-tree-right-side-view/
from collections import deque
class Solution:
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        if not root:
            return []

        res = []
        queue = deque([root])
        while queue:
            res.append(queue[-1].val)
            for _ in range(len(queue)):
                node = queue.popleft()
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)

        return res

# https://leetcode.com/problems/insert-into-a-sorted-circular-linked-list/
class Solution:
    def insert(self, head: 'Optional[Node]', insertVal: int) -> 'Node':
        newNode = Node(insertVal)
        if not head:
            newNode.next = newNode
            return newNode
        
        min = head
        max = head

        cur = head
        count = 0
        
        while True:
            count += 1
            if cur.val < min.val:
                min = cur
            if cur.val >= max.val:
                max = cur
            cur = cur.next
            if cur == head:
                break

        pre = max
        cur = min
        for _ in range(count):
            if cur.val > insertVal:
                break
            pre = cur
            cur = cur.next
            
        pre.next = newNode
        newNode.next = cur

        return head

# https://leetcode.com/problems/merge-intervals/
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort(key=lambda interval:interval[0])
        new_intervals = []
        for interval in intervals:
            if not new_intervals or interval[0] > new_intervals[-1][1]:
                new_intervals.append(interval)
            else:
                new_intervals[-1][1] = max(new_intervals[-1][1], interval[1])
        
        return new_intervals

# https://leetcode.com/problems/group-shifted-strings/
from collections import defaultdict
class Solution:
    def groupStrings(self, strings: List[str]) -> List[List[str]]:
        def hashKey(string):
            # shift all characters by `shift` number
            # imagine scale them on array of a -> z [0,...,25]
            # must floor 26, and then mod 26 to find the index after shifting
            shift = ord(string[0]) - ord('a')
            key = ''

            for c in string:
                key += chr((ord(c) - shift + 26) % 26)

            return key
        
        # ** S.2 **
        # def hashKey(string):
        #     shift = ord(string[0]) - ord('a')
        #     key = []

        #     for c in string:
        #         mapped = ord(c) - ord('a')
        #         located = (mapped - shift + 26) % 26
        #         key.append(str(located))

        #     return '.'.join(key)
        
        groups = defaultdict(list)

        for string in strings:
            key = hashKey(string)
            groups[key].append(string)

        return list(groups.values())

# https://leetcode.com/problems/maximum-swap/
# Intuition: largest possible number is when digits in decreasing order
class Solution:
    def maximumSwap(self, num: int) -> int:
        digits = []
        while num != 0:
            digits.append(num % 10)
            num = num // 10

        digits.reverse()
        n = len(digits)
        for i in range(n):
            max = i
            for j in range(i, n):
                if digits[j] >= digits[max]:
                    max = j

            if digits[max] > digits[i]:
                digits[max],digits[i] = digits[i], digits[max]
                break

        res = 0
        for digit in digits:
            res = res * 10 + digit
        return res

# https://leetcode.com/problems/diameter-of-binary-tree/
# calc diameter at each node => get max
# diameter = leftHeight + rightHeight
class Solution:
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        def getHeight(node):
            if not node:
                return 0
            leftHeight = getHeight(node.left)
            rightHeight = getHeight(node.right)

            self.ans = max(self.ans, leftHeight + rightHeight) 
            return max(leftHeight, rightHeight) + 1
        self.ans = 0
        getHeight(root)
        return self.ans


# https://leetcode.com/problems/exclusive-time-of-functions/
class Solution:
    def exclusiveTime(self, n: int, logs: List[str]) -> List[int]:
        ans = [0 for _ in range(n)]
        stack = []
        for log in logs:
            taskId, state, timestamp = log.split(':')
            if state == 'start':
                stack.append((taskId, timestamp))
            else:
                endTs = timestamp
                endedTaskId, startTs = stack.pop()
                assert endedTaskId == taskId
                time = int(endTs) - int(startTs) + 1
                ans[int(endedTaskId)] += time
                if stack:
                    ans[int(stack[-1][0])] -= time

        return ans

# https://leetcode.com/problems/diagonal-traverse/
# BFS level by level
class Solution:
    def findDiagonalOrder(self, mat: List[List[int]]) -> List[int]:
        m = len(mat)
        n = len(mat[0])

        ans = []
        leftToRight = True
        level = [(0,0)]

        while level:
            nextLevel = []
            for i in range(len(level)-1, -1, -1):
                x,y = level[i]
                ans.append(mat[x][y])

                newX = x+1 if leftToRight else x
                newY = y if leftToRight else y+1
                nextLevel.append((newX,newY))

                if i == 0: 
                    newX = x if leftToRight else x+1
                    newY = y+1 if leftToRight else y
                    nextLevel.append((newX,newY))
                        
            level = [(x, y) for x,y in nextLevel if x >= 0 and x <= m-1 and y >= 0 and y <= n-1]
            leftToRight = not leftToRight

        return ans