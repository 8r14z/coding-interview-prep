# h-ttps://www.teamblind.com/post/New-Year-Gift---Curated-List-of-Top-75-LeetCode-Questions-to-Save-Your-Time-OaM1orEU

# https://leetcode.com/problems/two-sum/
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        seen_value_indices = {}

        for index, num in enumerate(nums):
            prev_seen_value = target - num 
            if prev_seen_value in seen_value_indices:
                prev_seen_index = seen_value_indices[prev_seen_value]
                return [prev_seen_index, index]
            seen_value_indices[num] = index

        return []

# https://leetcode.com/problems/best-time-to-buy-and-sell-stock/
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        max_profit = 0
        min_price = float('inf')
        
        for price in prices:
            min_price = min(price, min_price)
            max_profit = max(max_profit, price - min_price)

        return max_profit

# https://leetcode.com/problems/contains-duplicate/
class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        existed = set()
        for num in nums:
            if num in existed:
                return True
            
            existed.add(num)
            
        return False

# https://leetcode.com/problems/product-of-array-except-self/
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        n = len(nums)
        result = [0] * n

        prefix = nums[0]
        for i in range(1, n):
            result[i] = prefix
            prefix *= nums[i]
        
        postfix = nums[n-1]
        for i in reversed(range(n-1)):
            if i == 0:
                result[0] = postfix
            else:
                result[i] *= postfix
                postfix *= nums[i]

        return result

# https://leetcode.com/problems/maximum-subarray/
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        subarray_sum = nums[0]
        result = subarray_sum
        for i in range(1, len(nums)):
            subarray_sum = max(subarray_sum + nums[i], nums[i])
            result = max(result, subarray_sum)

        return result

# https://leetcode.com/problems/maximum-product-subarray/
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        min_subarray = nums[0]
        max_subarray = nums[0]
        result = nums[0]

        for i in range(1, len(nums)):
            new_max_subarray = max(max_subarray * nums[i], min_subarray * nums[i], nums[i])
            new_min_subarray = min(max_subarray * nums[i], min_subarray * nums[i], nums[i])
            result = max(result, new_max_subarray)

            max_subarray = new_max_subarray
            min_subarray = new_min_subarray

        return result

# https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/
class Solution:
    def findMin(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 1 or nums[0] < nums[-1]:
            return nums[0]
        
        left = 0 
        right = len(nums)-1
        min_idx = right
        while left <= right: 
            mid = (left + right) // 2
            if nums[mid] >= nums[-1]:
                left = mid + 1
            else:
                min_idx = min(min_idx, mid)
                right = mid - 1
        
        return nums[min_idx]

# https://leetcode.com/problems/search-in-rotated-sorted-array/
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        n = len(nums)
        if n == 0:
            return -1
        elif n == 1:
            return 0 if nums[0] == target else -1
            
        def find_min_index():
            low = 0
            high = n - 1
            index = high
            while low <= high:
                mid = (low + high) // 2
                if nums[mid] >= nums[0]:
                    low = mid + 1 
                else:
                    index = min(index,mid)
                    high = mid - 1
            return index
        
        def find_target_index(low, high):
            while low <= high:
                mid = (low + high) // 2
                if nums[mid] == target:
                    return mid
                elif nums[mid] > target:
                    high = mid - 1
                else:
                    low = mid + 1
            return -1
        
        min_index = 0 if nums[0] < nums[-1] else find_min_index()
        if min_index == 0:
            return find_target_index(0, n-1)
        else:
            index = find_target_index(0, min_index - 1)
            if index == -1:
                index = find_target_index(min_index, n-1)
            return index

# S.2
# Intuition: if low value <= mid, left portion is sorted, try search target within left portion, otherwise search right portion
# else right portion is merged, ...
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        n = len(nums)
        low = 0
        high = n-1

        while low <= high:
            mid = (low + high) // 2
            if nums[mid] == target:
                return mid

            if nums[low] <= nums[mid]: # left portion is merged
                if nums[low] <= target and target <= nums[mid]:
                    high = mid - 1
                else:
                    low = mid + 1
            else:
                if nums[mid] <= target and target <= nums[high]:
                    low = mid + 1
                else:
                    high = mid - 1
                    
        return -1
            
# https://leetcode.com/problems/3sum/
## S.1 - hash table 
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        triplets = set()
        n = len(nums)
        if n < 3:
            return triplets
        
        for i in range(n):
            target = -nums[i]
            temp = set()
            for j in range(i+1, n):
                if target - nums[j] in temp:
                    triplet = tuple(sorted([nums[i], target - nums[j], nums[j]]))
                    triplets.add(triplet)
                temp.add(nums[j])
        
        return triplets
## S.2 - two pointers - O(n^2) less constant factor
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        n = len(nums)
        triplets = []
        for i in range(n):
            if i > 0 and nums[i] == nums[i-1]:
                continue
                
            left = i+1
            right = n-1
            while left < right:
                target3sum = nums[i] + nums[left] + nums[right]
                if target3sum == 0:
                    triplet = [nums[i], nums[left], nums[right]]
                    triplets.append(triplet)
                    left += 1
                    while nums[left] == nums[left-1] and left < right:
                        left += 1
                elif target3sum > 0:
                    right -= 1
                else:
                    left += 1
                    
        return triplets

# https://leetcode.com/problems/container-with-most-water/
# Greedy:
# Start with the maximum width container and go to a shorter width container 
# if there is a vertical line longer than the current containers shorter line. 
# This way we are compromising on the width but we are looking forward to a longer length container.
class Solution:
    def maxArea(self, height: List[int]) -> int:
        n = len(height)
        left = 0
        right = n - 1
        res = -1
        while left < right:
            if height[left] < height[right]:
                res = max(height[left] * (right - left), res)
                left += 1
            else:
                res = max(height[right] * (right - left), res)
                right -= 1
                
        return res

# https://leetcode.com/problems/sum-of-two-integers/
# h-ttps://leetcode.com/problems/sum-of-two-integers/solutions/132479/simple-explanation-on-how-to-arrive-at-the-solution/
# h-ttps://leetcode.com/problems/sum-of-two-integers/solutions/489210/read-this-if-you-want-to-learn-about-masks/
class Solution:
    def getSum(self, a: int, b: int) -> int:
        mask = 0xffffffff # 4 bytes

        while (b & mask) != 0:
            carry = ( a & b ) << 1
            a = (a ^ b) 
            b = carry
        
        return (a & mask) if b > 0 else a

# https://leetcode.com/problems/number-of-1-bits/
class Solution:
    def hammingWeight(self, n: int) -> int:
        count = 0
        while n:
            count += (n & 1)
            n >>= 1
        return count

# https://leetcode.com/problems/counting-bits/
class Solution:
    def countBits(self, n: int) -> List[int]:
        res = [0]
        for i in range(1, n+1):
            # // 2 is equal to shift right by 1 (>> 1)
            # * 2 is equal to shift left by 1 (<< 1)
            # => same number of bits
            # if odd number need to retain 1 LSB as it gets lost while shifting right
            res.append(res[i//2] + i % 2)
            
        return res

# https://leetcode.com/problems/missing-number/
class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        n = len(nums)
        total = n * (n+1) // 2
        return total - sum(nums)

# https://leetcode.com/problems/reverse-bits/
class Solution:
    def reverseBits(self, n: int) -> int:
        res = 0
        for i in range (32):
            bit = n % 2
            res += (bit * 2 ** (32-i-1))
            n >>= 1
        return res

# https://leetcode.com/problems/climbing-stairs/
# similar to fibonacci
class Solution:
    def climbStairs(self, n: int) -> int:
        if n == 1:
            return 1
        elif n == 2:
            return 2
        first = 1
        second = 2
        
        for i in range(3, n):
            cur = first + second
            first = second
            second = cur
        
        return first + second

# https://leetcode.com/problems/coin-change/
# DP variant
# To solve this, we need to ask question what will be minimum no coins at amount - 1, amount - 2, ..., 1
# And at each amount of money, we will need to question whether we should include a coin in [coins] or not
#
# O(n * amount)
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        n = len(coins)
        if n == 0 or amount == 0:
            return 0

        min_coins_at_amount = [0]

        for cur_amount in range(1, amount+1):
            min_coins = float('inf')
            for coin in coins:
                if coin <= cur_amount:
                    min_coins = min(min_coins, min_coins_at_amount[cur_amount - coin] + 1)
            min_coins_at_amount.append(min_coins)

        return min_coins_at_amount[amount] if min_coins_at_amount[amount] != float('inf') else -1

# https://leetcode.com/problems/longest-increasing-subsequence/
# [../leetcode/300_longest_increasing_subseq.py] for DP solution
import bisect
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        n = len(nums)
        sequence = [nums[0]]
        for i in range(1, n):
            if nums[i] > sequence[-1]:
                sequence.append(nums[i])
            else:
                index = bisect.bisect_left(sequence, nums[i])
                sequence[index] = nums[i]
        return len(sequence)
        
# https://leetcode.com/problems/word-break/
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        words = set(wordDict)
        n = len(s)
        is_valid_word = [False] * n
        
        for i in reversed(range(n)):
            for j in range(i, n):
                word = s[i:j+1]
                if word in words and (j == n-1 or is_valid_word[j+1]):
                    is_valid_word[i] = True
                    break
        
        return is_valid_word[0]

# https://leetcode.com/problems/combination-sum-iv/
# the question to ask here is what the combination of target-1, target-2,..., 0 is
# O(target * n)
# num of sub-problems = target
# cost / sub-problem = n
from functools import cache
class Solution:
    def combinationSum4(self, nums: List[int], target: int) -> int:
        @cache 
        def dp(n):
            if n == 0:
                return 1
            res = 0
            for num in nums:
                if n >= num:
                    res += dp(n-num)
            return res
            
        return dp(target)
    
    def combinationSum4(self, nums: List[int], target: int) -> int:
        dp = [0] *(target+1)
        dp[0] = 1
        
        for cur_target in range(1, target+1):
            for num in nums:
                if cur_target >= num:
                    dp[cur_target] += dp[cur_target-num]
                    
        return dp[target]

# https://leetcode.com/problems/house-robber/
# there is a little trick in the question that makes u think it always a sequence of i, i + 2, i + 4, ...
# it could be any house as long as its index < i-1 or > i+1
# lession learned: don't assume the input :) 
# dp question: whether max is sum of prev house or cur + prev_prev housee
# O(n), O(1) space
class Solution:
    def rob(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        if n == 1:
            return nums[0]
        if n == 2:
            return max(nums[0], nums[1])
        
        max_prev_prev_house = nums[0]
        max_prev_house = max(nums[1], nums[0])
        max_cur = 0
        
        for i in range(2, n):
            max_cur = max(max_prev_house, max_prev_prev_house + nums[i])
            max_prev_prev_house = max_prev_house
            max_prev_house = max_cur
        return max_cur

# https://leetcode.com/problems/house-robber-ii/
# there are 2 sub-prolems, what is max money includes/excludes first house? 
class Solution:
    def rob(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        elif n == 1:
            return nums[0]
        elif n == 2:
            return max(nums[0], nums[1])
            
        max_prev_prev_house = nums[0]
        max_prev_house = max(nums[0], nums[1])
        
        max_prev_prev_house_ex_1 = 0
        max_prev_house_ex_1 = nums[1]
        
        for i in range(2, n-1):
            max_cur = max(max_prev_house, max_prev_prev_house + nums[i])
            max_prev_prev_house = max_prev_house
            max_prev_house = max_cur
            
            max_cur_ex_1 = max(max_prev_house_ex_1, max_prev_prev_house_ex_1 + nums[i])
            max_prev_prev_house_ex_1 = max_prev_house_ex_1
            max_prev_house_ex_1 = max_cur_ex_1
        
        return max(max_prev_house, max_prev_prev_house_ex_1 + nums[-1])

# https://leetcode.com/problems/decode-ways/
# question is how many ways we can reach the current char
# the current char can be single digit or 2nd digit of prev char
# so cur_ways = prev_ways + prev_prev_ways
# it's kinda similar to staircases problem (number of ways to reach stair x)
# h-ttps://leetcode.com/problems/climbing-stairs/
class Solution:
    def numDecodings(self, s: str) -> int:
        n = len(s)
        memo = [0] * n
        memo[0] = 1 if s[0] != '0' else 0

        for i in range(1, n):
            c = s[i]
            if c != '0': # if c == '0' it can NOT stand alone
                memo[i] = memo[i-1]
            
            pre = s[i-1]
            if (pre == '1' and c <= '9') or (pre == '2' and c <= '6'):
                memo[i] += (memo[i-2] if i >= 2 else 1)

        return memo[-1]

# space optimal sol        
class Solution:
    def numDecodings(self, s: str) -> int:
        n = len(s)
        prev_prev = 0
        prev = 1 if s[0] != '0' else 0
        
        for i in range(1, n):
            char = s[i]
            ways = 0
            if char != '0': # char can be a single digit alone. e.g 2
                ways += prev 
                
            # OR it can be the 2nd digit of previous char. e.g 12 
            prev_char = s[i-1]
            if (prev_char == '1' and char <= '9') or (prev_char == '2' and char <= '6'):
                ways += (1 if i-1 == 0 else prev_prev)
                
            prev_prev = prev
            prev = ways
        
        return prev
            
# https://leetcode.com/problems/unique-paths/
# fairly simple problem
# dp table to track num of ways to reach position [i][j]
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:        
        dp = [[0] * (n) for _ in range(m)]
            
        for i in range(0, m):
            for j in range(0, n):
                if i == 0 or j == 0:
                    dp[i][j] = 1
                    continue 
                dp[i][j] = dp[i-1][j] + dp[i][j-1]
        
        return dp[m-1][n-1]
        
# https://leetcode.com/problems/jump-game/
# keep track of the min index where current position can jump to
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        n = len(nums)
        if n == 0:
            return False
        if n == 1:
            return True
        
        min_jumpable_index = n-1

        for i in reversed(range(n-1)):
            if i + nums[i] >= min_jumpable_index:
                min_jumpable_index = i
        
        return min_jumpable_index == 0

# https://leetcode.com/problems/clone-graph/
# DFS or BFS is fine
# Used DFS in the solution since it cosumes less memory
class Node:
    def __init__(self, val = 0, neighbors = None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []
class Solution:
    def cloneGraph(self, node: 'Node') -> 'Node':
        mapped_nodes = {}

        def clone_node(node):
            if not node:
                return None

            new_node = Node(node.val, [] if node.neighbors else None)
            mapped_nodes[node] = new_node
            
            for neighbor in node.neighbors:
                if neighbor not in mapped_nodes:
                    new_neighbor = clone_node(neighbor)
                    new_node.neighbors.append(new_neighbor)
                else:
                    new_node.neighbors.append(mapped_nodes[neighbor])
                
            return new_node
            
            
        return clone_node(node)

# https://leetcode.com/problems/course-schedule/
# DFS to find cyclic graph
from collections import defaultdict
class Solution:
    def __init__(self):
        self.visited = set()
        self.finished = set()
        self.graph = defaultdict(list)

    def found_cycle_at_node(self, node):
        self.visited.add(node)

        for neighbor in self.graph[node]:
            if neighbor not in self.visited:
                if self.found_cycle_at_node(neighbor):
                    return True
            elif neighbor not in self.finished:
                return True

        self.finished.add(node)
        return False
        
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        for edge in prerequisites:
            p,q = edge
            self.graph[p].append(q)

        for i in range(numCourses):
            if self.found_cycle_at_node(i):
                return False

        return True

# https://leetcode.com/problems/pacific-atlantic-water-flow/
# dfs from edges and hold 2 set of statuses if a node can reach alantic or pacific
# the result is alantic[i][j] and pacific[i][j]
from collections import deque
class Solution:
    def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:
        m = len(heights)
        n = len(heights[0])
        if m < 1 and n < 1:
            return []
        
        def neighbors(i, j):
            neighbors = []
            height = heights[i][j]
            if i > 0 and height <= heights[i-1][j]:
                neighbors.append([i-1, j])
            if i < m-1 and height <= heights[i+1][j]:
                neighbors.append([i+1, j])
            if j > 0 and height <= heights[i][j-1]:
                neighbors.append([i, j-1])
            if j < n-1 and height <= heights[i][j+1]:
                neighbors.append([i, j+1])
            return neighbors

        def dfs(i, j, visited):
            visited[i][j] = True
            for ni,nj in neighbors(i,j):
                if not visited[ni][nj]:
                    dfs(ni, nj, visited)

        pacific_visited = [[False] * n for _ in range(m)]
        atlantic_visited = [[False] * n for _ in range(m)]
        
        for i in range(m):
            dfs(i, 0, pacific_visited)
            dfs(i, n-1, atlantic_visited)
        
        for i in range(n):
            dfs(0, i, pacific_visited)
            dfs(m-1, i, atlantic_visited)
        
        result = []
        for i in range(m):
            for j in range (n):
                if pacific_visited[i][j] and atlantic_visited[i][j]:
                    result.append([i, j])
                    
        return result

# https://leetcode.com/problems/number-of-islands
# DFS - space optimization by reuse the grid :)
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        n = len(grid)
        m = len(grid[0])

        def dfs(i, j, visited):
            if i < 0 or i >= n or j < 0 or j >= m or grid[i][j] == '0':
                return    
            
            if grid[i][j] == '#':
                return

            grid[i][j] = '#'
            dfs(i+1, j, visited)
            dfs(i-1, j, visited)
            dfs(i, j+1, visited)
            dfs(i, j-1, visited)
            
        visited = set()
        ans = 0
        for i in range(n):
            for j in range(m):
                if grid[i][j] == '1':
                    dfs(i,j, visited)
                    ans += 1
        return ans

# https://leetcode.com/problems/longest-consecutive-sequence/
# keep all num in a set for quick access
# the sequence is started when a num-1 not available <= meaning it's the min num in the sequence
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        if not nums:
            return 0
        numset = set(nums)
        ans = -float('inf')
        for num in nums:
            if num - 1 not in numset: # start of sequence
                length = 1
                while num + length in numset:
                    length += 1
                ans = max(ans, length)
        return ans
# Sol2 use Union-Find
# Union Find implementation in (./source/data_structure/union_find.py)
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        if not nums:
            return 0       
        
        parent = {}
        
        def find(x):
            if x not in parent:
                return x
            rootx = find(parent[x])
            parent[x] = rootx
            return rootx
        
        def union(u, v):
            parent[find(v)] = find(u)
                
        numset = set(nums)
        for n in numset:
            if n-1 in numset:
                union(n-1, n)

        res = defaultdict(int)
        for v in numset:
            v = find(v)
            res[v] += 1
        return max(res.values())

# https://leetcode.com/problems/alien-dictionary/
class Solution:
    def alienOrder(self, words: List[str]) -> str:
        adjencency_list = defaultdict(set)
        in_degree = {c: 0 for word in words for c in word}
    
        for first_word, second_word in zip(words, words[1:]):
            for c, d in zip(first_word, second_word):
                if c != d: 
                    if d not in adjencency_list[c]:
                        adjencency_list[c].add(d)
                        in_degree[d] += 1
                    break
            else:
                if len(second_word) < len(first_word): 
                    return ""
            
        visited = set()
        queue = deque([c for c in in_degree if in_degree[c] == 0])
        
        output = []
        
        while queue:
            node = queue.popleft()
            output.append(node)
            visited.add(node)
            for child in adjencency_list[node]:
                in_degree[child] -= 1
                if not in_degree[child]:
                    queue.append(child)
        
        if len(output) < len(in_degree):
            return ""
    
        return ''.join(output)

# https://leetcode.com/problems/graph-valid-tree/
# LC Premium - no access :(

# https://leetcode.com/problems/number-of-connected-components-in-an-undirected-graph/
# LC Premium as well :(

# https://leetcode.com/problems/insert-interval/
START = 0
END = 1
class Solution:
    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        index = bisect.bisect_left(intervals, newInterval[0], key=lambda tup:tup[0])
        intervals.insert(index, newInterval)
        
        new_intervals = []
        for interval in intervals:
            if not new_intervals or interval[START] > new_intervals[-1][END]:
                new_intervals.append(interval)
            else:
                new_intervals[-1][END] = max(new_intervals[-1][END], interval[END])
    
        return new_intervals

# https://leetcode.com/problems/merge-intervals/
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort(key=lambda interval:(interval[0], interval[1]))
        new_intervals = []
        for interval in intervals:
            if not new_intervals or interval[START] > new_intervals[-1][END]:
                new_intervals.append(interval)
            else:
                new_intervals[-1][END] = max(new_intervals[-1][END], interval[END])
        
        return new_intervals

# https://leetcode.com/problems/non-overlapping-intervals/
# Greedily to find the max number of intervals can be execute -> min removals
# Equal to the time scheduling problem to find the max number of tasks :) 
# h-ttps://en.wikipedia.org/wiki/Interval_scheduling
# Selecting the intervals that start earliest is not an optimal solution, because if the earliest interval happens to be very long, accepting it would make us reject many other shorter requests.
# Selecting the shortest intervals or selecting intervals with the fewest conflicts is also not optimal.
# [Earliest deadline first scheduling] The queue will be searched for the process closest to its deadline
class Solution:
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        intervals.sort(key=lambda interval:interval[END])

        ans = 0
        cur = None
        for interval in intervals:
            if cur is None or interval[START] >= cur[END]:
                cur = interval
            else:
                ans += 1
        
        return ans

# https://leetcode.com/problems/meeting-rooms/
# LC Premium

# https://leetcode.com/problems/meeting-rooms-ii/
# LC Premium

# https://leetcode.com/problems/reverse-linked-list/
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev = None
        cur = head
        while cur:
            next = cur.next
            cur.next = prev
            prev = cur
            cur = next
            
        return prev

# https://leetcode.com/problems/linked-list-cycle/
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        if not head:
            return False
        
        slow = fast = head
        while fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow is fast:
                return True
            
        return False

# https://leetcode.com/problems/merge-two-sorted-lists/
class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        if not list1:
            return list2
        if not list2:
            return list1
        
        if list1.val <= list2.val:
            list1.next = self.mergeTwoLists(list1.next, list2)
            return list1
        else:
            list2.next = self.mergeTwoLists(list1, list2.next)
            return list2

# https://leetcode.com/problems/merge-k-sorted-lists/
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
import heapq
class CompareableNode:
    def __init__(self, node: ListNode = None):
        self.node = node
    
    def __lt__(self, other):
        return self.node.val < other.node.val

class Solution(object):
    def mergeKLists(self, lists):
        min_heap = []
        for head in lists:
            if head:
                heapq.heappush(min_heap, CompareableNode(head))
        
        if not min_heap:
            return None
        
        head = min_heap[0].node
        cur = None
        while min_heap:
            min_node = heapq.heappop(min_heap).node
            if cur:
                cur.next = min_node
            cur = min_node
            if cur.next:
                heapq.heappush(min_heap, CompareableNode(cur.next))
                
        return head

# https://leetcode.com/problems/remove-nth-node-from-end-of-list/
class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        first = head
        for _ in range(n):
            first = first.next
        
        if not first:
            return head.next
        
        second = head
        while first.next:
            first = first.next
            second = second.next
        second.next = second.next.next
        return head
        
# https://leetcode.com/problems/reorder-list/
class Solution:
    def reorderList(self, head: Optional[ListNode]) -> None:
        if not head:
            return
        
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next

        def reverse(head):
            prev = None
            cur = head
            while cur:
                next = cur.next
                cur.next = prev
                prev = cur
                cur = next
            return prev
                
        tail = reverse(slow)
        
        while tail.next: # the last node in head ll is still linked to last node in tail ll
            head_next = head.next
            tail_next = tail.next
            head.next = tail
            tail.next = head_next
            head = head_next
            tail = tail_next
            
# https://leetcode.com/problems/set-matrix-zeroes/
# use first row and first col to store the state if there is a zero found in that col/row
class Solution:
    def setZeroes(self, matrix: List[List[int]]) -> None:
        m = len(matrix)
        n = len(matrix[0])
        
        has_zero = False
        
        for i in range(m):
            if matrix[i][0] == 0:
                has_zero = True
            for j in range(1,n):
                if matrix[i][j] == 0:
                    matrix[i][0] = matrix[0][j] = 0
                    
        for i in range(1,m):
            for j in range(1,n):
                if matrix[i][0] == 0 or matrix[0][j] == 0:
                    matrix[i][j] = 0
        
        if matrix[0][0] == 0:
            for j in range(n):
                matrix[0][j] = 0
                
        if has_zero:
            for i in range(m):
                matrix[i][0] = 0

# https://leetcode.com/problems/spiral-matrix/
# just follow the order
# nothing hard but it's a little bit frustrating with repeating items added -> use count < total to prevet that
class Solution:
    def spiralOrder(self, matrix):
        row = len(matrix)
        col = len(matrix[0])
        total = row*col
        count = 0

        start_row = start_col = 0
        end_row = row-1
        end_col = col-1

        result = []

        while count < total:
            for j in range(start_col, end_col+1):
                if count < total:
                    count += 1
                    result.append(matrix[start_row][j])
            start_row += 1

            for i in range(start_row, end_row+1):
                if count < total:
                    count += 1
                    result.append(matrix[i][end_col])
            end_col -= 1
            
            for j in range(end_col, start_col-1, -1):
                if count < total:
                    count += 1
                    result.append(matrix[end_row][j])
            end_row -= 1

            for i in range(end_row, start_row-1, -1):
                if count < total:
                    count += 1
                    result.append(matrix[i][start_col])
            start_col += 1

        return result

# https://leetcode.com/problems/rotate-image/
class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        n = len(matrix)
        
        for i in range(n):
            for j in range(i+1, n):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
                
        for i in range(n):
            for j in range(n//2):
                matrix[i][j], matrix[i][n-1-j] = matrix[i][n-1-j], matrix[i][j]

# https://leetcode.com/problems/word-search/
# Just use DFS to find the word
# One trick to save extra memory is to modify the board content (e.g to #)
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        l = len(word)
        n = len(board)
        m = len(board[0])
        VISITED = '#'
        
        def dfs(i, j, wi):
            if wi == l:
                return True
            
            if i >= n or i < 0:
                return False
            if j >= m or j < 0:
                return False
            if board[i][j] == VISITED:
                return False
            if board[i][j] != word[wi]:
                return False
            
            board[i][j] = VISITED
            found = dfs(i+1, j, wi+1) or dfs(i-1, j, wi+1) or dfs(i, j+1, wi+1) or dfs(i, j-1, wi+1)
            board[i][j] = word[wi]
            return found
        
        for i in range(n):
            for j in range(m):
                if dfs(i, j, 0):
                    return True
        
        return False

# https://leetcode.com/problems/longest-substring-without-repeating-characters/
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        n = len(s)
        if n < 1: return 0

        res = float('-inf')
        previous_index = {}
        start = 0
        for end, c in enumerate(s):
            if c in previous_index and previous_index[c] >= start: # repeating char can be before start
                start = previous_index[c] + 1

            previous_index[c] = end
            cur_non_repeating_length = end-start+1
            res = max(res, cur_non_repeating_length)

        return res

# https://leetcode.com/problems/longest-repeating-character-replacement/
# sliding windown, one condition to keep track is the lenght of substring - most repeating char count <= k
class Solution:
    def characterReplacement(self, s: str, k: int) -> int:
        answer = float('-inf')
        
        char_count = [0] * 26
        start = 0
        for i in range(len(s)):
            c = s[i]
            char_count[ord(c) - ord('A')] += 1
            max_repeating_count = max(char_count)
            substring_length = i + 1 - start
            
            if substring_length - max_repeating_count > k: # can NOT replace 
                char_count[ord(s[start]) - ord('A')] -= 1
                substring_length -= 1
            
            start = i + 1 - substring_length
            answer = max(answer, substring_length)

        return answer

# https://leetcode.com/problems/minimum-window-substring/
# ---

# https://leetcode.com/problems/valid-anagram/
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        if len(s) != len(t):
            return False
        
        chars = [0] * 26
        for c in s:
            chars[ord(c) - ord('a')] += 1
        
        for c in t:
            chars[ord(c) - ord('a')] -= 1
            
        return max(chars) == 0

# https://leetcode.com/problems/group-anagrams/
# if a char only appears once in any string, we can use binary mask :) 
from collections import defaultdict
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        map = defaultdict(list)

        for s in strs:
            mask = [0] * 26
            for c in s:
                mask[ord(c) - ord('a')] += 1
            map[tuple(mask)].append(s)
            
        return map.values()

# https://leetcode.com/problems/valid-parentheses/
class Solution:
    def isValid(self, s: str) -> bool:
        stack = []
        
        for c in s:
            if not stack:
                stack.append(c)
                continue
            
            if ((c == ')' and stack[-1] == '(') 
                or (c == '}' and stack[-1] == '{')
                or (c == ']' and stack[-1] == '[')):
                stack.pop()
            else:
                stack.append(c)
        
        return len(stack) == 0

# https://leetcode.com/problems/valid-palindrome/
# O(1) space
class Solution:
    def isPalindrome(self, s: str) -> bool:
        left = 0
        right = len(s)-1
        while left < right:
            if not s[left].isalnum() and not s[right].isalnum():
                left += 1
                right -= 1
            elif not s[left].isalnum():
                left += 1
            elif not s[right].isalnum():
                right -= 1
            elif s[left].lower() != s[right].lower():
                return False
            else:
                left += 1
                right -= 1
        return True

# https://leetcode.com/problems/longest-palindromic-substring/
class Solution:
    def longestPalindrome(self, s: str) -> str:
        def palindrome(s, l, r):
            while l >= 0 and r < len(s) and s[l] == s[r]:
                l -= 1
                r += 1
            return r - l - 1
        
        start = end = 0
        for i in range(len(s)):
            l1 = palindrome(s, i, i)
            l2 = palindrome(s, i, i+1)
            l = max(l1,l2)
            if l > (end-start+1):
                start = i - (l-1)//2
                end = i + l//2
                
        return s[start:end+1]

# https://leetcode.com/problems/palindromic-substrings/
class Solution:
    def countSubstrings(self, s: str) -> int:
        def count_palindrome_in_range(i, j):
            answer = 0
            while i >= 0 and j < len(s) and s[i] == s[j]:
                i -= 1
                j += 1
                answer += 1

            return answer

        s_len = len(s)
        answer = 0

        for i in range(s_len):
            odd_len_count = count_palindrome_in_range(i, i)
            even_len_count = count_palindrome_in_range(i, i+1)
            answer += (odd_len_count + even_len_count)

        return answer

# https://leetcode.com/problems/encode-and-decode-strings/
# LC Premium

# https://leetcode.com/problems/maximum-depth-of-binary-tree/
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
        
        stack = [root]
        depth = {root: 1}
        res = 0
        while stack: 
            node = stack.pop()
            
            left = node.left
            right = node.right
            if not left and not right:
                res = max(res, depth[node])
                
            if left and left not in depth:
                stack.append(left)
                depth[left] = 1 + depth[node]
                
            if right and right not in depth:
                stack.append(right)
                depth[right] = 1 + depth[node]
                
        return res
    
# https://leetcode.com/problems/same-tree/
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        if not p and not q:
            return True
        elif not p or not q:
            return False 

        return p.val == q.val and self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right) 
    
# https://leetcode.com/problems/invert-binary-tree/
# Solve recursively or iteration as following
class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if not root: 
            return None

        queue = deque([root])

        while queue:
            node = queue.popleft()
            left = node.left
            right = node.right
            node.left = right
            node.right = left 
            
            if left:
                queue.append(left)
            if right:
                queue.append(right)

        return root

# https://leetcode.com/problems/binary-tree-maximum-path-sum/
# --- 

# https://leetcode.com/problems/binary-tree-level-order-traversal/
class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        result = []
        
        if not root:
            return result
        
        queue = deque([root])
        
        while queue:
            cur_level_count = len(queue)
            cur_level = []

            for _ in range(cur_level_count):
                node = queue.popleft()
                cur_level.append(node.val)
                
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)

            result.append(cur_level)

        return result
    
# https://leetcode.com/problems/serialize-and-deserialize-binary-tree/
# ---

# https://leetcode.com/problems/subtree-of-another-tree/
# Time Complexity is O(m*n) with m and n are size of corresponding trees
# For each node of root we run `isSameTree`
# 
# An optimization is to serialize trees into strings, then use KMP or Rabin-Karp (hashing) to search substring
class Solution:
    def isSubtree(self, root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:
        def isSameTree(p, q):
            if not p and not q:
                return True
            elif not p or not q:
                return False

            return p.val == q.val and isSameTree(p.left, q.left) and isSameTree(p.right, q.right)
        
        if not subRoot: 
            return True
        
        if not root:
            return False

        return isSameTree(root, subRoot) or self.isSubtree(root.left, subRoot) or self.isSubtree(root.right, subRoot)

# https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/
# unique value
# Intuition: 
# preorder travels the tree in Node - Left - Right order
# inorder travels the tree in Left - Node - Right order
# preorder tree will let us pick the root node of the tree/substree
# The root's left child is in left subarray of inorder array [start, root_index - 1]
# The root's right child is in right subarray in inorder array [root_index + 1, end]
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        n = len(preorder)
        self.cur_root_index = 0

        inorder_node_indicies = {}
        for i, v in enumerate(inorder):
            inorder_node_indicies[v] = i

        def build_tree(start, end):
            if start > end:
                return None
            
            cur_val = preorder[self.cur_root_index]
            self.cur_root_index += 1

            root = TreeNode(val = cur_val)

            if start == end: # no children, inorder array one item
                return root

            cur_inorder_index = inorder_node_indicies[cur_val]
            root.left = build_tree(start, cur_inorder_index - 1)
            root.right = build_tree(cur_inorder_index + 1, end)

            return root

        return build_tree(0, n-1)


# https://leetcode.com/problems/validate-binary-search-tree/
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
INF = float('inf')
class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        def is_valid(node, min, max) -> bool:
            if not node:
                return True
            
            if node.val <= min or node.val >= max:
                return False
            
            return is_valid(node.left, min, node.val) and is_valid(node.right, node.val, max)
            
        return is_valid(root, -INF, INF)

# https://leetcode.com/problems/kth-smallest-element-in-a-bst/
class Solution:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        stack = [root]
        visited = set([root])
        count = 0

        while stack:
            left = stack[-1].left
            if left and left not in visited:
                stack.append(left)
                continue

            node = stack.pop()
            visited.add(node)

            count += 1
            if count == k:
                return node.val
            
            right = node.right
            if right and right not in visited:
                stack.append(right)
            
        return None
    

# https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/
# Intuition: it's BST so all left nodes are less than root, all right nodes are greater than root
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if not root:
            return None

        if root.val > p.val and root.val > q.val:
            return self.lowestCommonAncestor(root.left, p, q)
        elif root.val < p.val and root.val < q.val:
            return self.lowestCommonAncestor(root.right, p, q)
        else:
            return root
        
# https://leetcode.com/problems/implement-trie-prefix-tree/
class Node:
    def __init__(self):
        self.children = {}
        self.word_end = False

class Trie:
    def __init__(self):
        self.root = Node()

    def insert(self, word: str) -> None:
        root = self.root
        for c in word:
            if c not in root.children:
                root.children[c] = Node()
            root = root.children[c]
        
        root.word_end = True

    def search(self, word: str) -> bool:
        root = self.root
        for c in word:
            if c not in root.children:
                return False
            root = root.children[c]

        return root.word_end

    def startsWith(self, prefix: str) -> bool:
        root = self.root
        for c in prefix:
            if c not in root.children:
                return False
            root = root.children[c]
        
        return True

# https://leetcode.com/problems/design-add-and-search-words-data-structure/
# Query trie node, if char == '.', query all children nodes
# Time: O(n)
class Node:
    def __init__(self):
        self.children = {}
        self.word_end = False

class WordDictionary:

    def __init__(self):
        self.root = Node()
        
    def addWord(self, word: str) -> None:
        root = self.root
        for c in word:
            if c not in root.children:
                root.children[c] = Node()
            root = root.children[c]
        
        root.word_end = True

    def search(self, word: str) -> bool:
        return self.__match(0, word, self.root)

    def __match(self, i, word, node) -> bool:
        for j in range(i, len(word)):
            char = word[j]
            if char == '.':
                for child in node.children.values():
                    if self.__match(j+1, word, child):
                        return True
                return False
            else:
                if char not in node.children:
                    return False
                node = node.children[char]
            
        return node.word_end

# https://leetcode.com/problems/word-search-ii/
# ---

#  https://leetcode.com/problems/merge-k-sorted-lists/
# --duplicated--

# https://leetcode.com/problems/top-k-frequent-elements/
from collections import Counter
class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        counter = Counter(nums)
        most_frequent = counter.most_common(k)
        return [value for value, _ in most_frequent]

# S.2 use max heap
from collections import defaultdict
import heapq

class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        frequencies = defaultdict(int)
        for num in nums:
            frequencies[num] += 1

        max_heap = []
        for num, count in frequencies.items():
            heapq.heappush(max_heap, (-1 * count, num))
        
        result = []
        for _ in range(k):
            count, num = heapq.heappop(max_heap)
            result.append(num)

        return result
    
# https://leetcode.com/problems/find-median-from-data-stream/
# ---