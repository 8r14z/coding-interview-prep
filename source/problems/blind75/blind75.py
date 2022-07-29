# https://www.teamblind.com/post/New-Year-Gift---Curated-List-of-Top-75-LeetCode-Questions-to-Save-Your-Time-OaM1orEU

# https://leetcode.com/problems/two-sum/
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        temp = {} 
        for (i, val) in enumerate(nums):
            num = target - val
            if num in temp:
                return [temp[num], i]
            temp[val] = i
            
        return []


# https://leetcode.com/problems/best-time-to-buy-and-sell-stock/
INF = float('inf')
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        n = len(prices)
        max_profit = 0
        max_price = -INF
        
        for i in reversed(range(n)):
            if (max_price >= prices[i]):
                cur_profit = max_price - prices[i]
                max_profit = cur_profit if cur_profit > max_profit else max_profit
            else:
                max_price = prices[i]
                
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
        if (n <= 1):
            return nums
        
        ans = [0] * n
        
        prefix = nums[0]
        for i in range(1, n):
            ans[i] = prefix
            prefix *= nums[i]
        
        postfix = nums[-1]
        for i in reversed(range(n-1)):
            ans[i] = postfix if i == 0 else ans[i] * postfix
            postfix *= nums[i]
        
        return ans

# https://leetcode.com/problems/maximum-subarray/
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        n = len(nums)
        max_so_far = 0
        res = -float('inf')
        for i in range(n):
            max_so_far = max(max_so_far + nums[i], nums[i])
            res = max(res, max_so_far)
            
        return res

# https://leetcode.com/problems/maximum-product-subarray/
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        n = len(nums)
        if n < 2:
            return nums[0]
        
        max_product_so_far = min_product_so_far = 1
        result = -float('inf')
        
        for i in range(n):
            cur_num = nums[i]
            local_max_product = max(max_product_so_far * cur_num, min_product_so_far * cur_num, cur_num)
            local_min_product = min(max_product_so_far * cur_num, min_product_so_far * cur_num, cur_num)
            
            max_product_so_far = local_max_product
            min_product_so_far = local_min_product
            
            result = max(max_product_so_far, result)

        return result

# https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/
class Solution:
    def findMin(self, nums: List[int]) -> int:
        first = nums[0]
        last = nums[-1]
        if first < last or len(nums) == 1:
            return first
        
        def search_min():
            n = len(nums)
            low = 0
            high = n-1
            min_idx = high
            while low <= high:
                mid = (low + high) // 2
                if nums[mid] >= first:
                    low = mid + 1
                else:
                    min_idx = min(min_idx, mid)
                    high = mid - 1
                    
            return nums[min_idx]
        
        return search_min()

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
class Solution:
    def getSum(self, a, b):
        if b == 0: return a
        return self.getSum(a ^ b, (a & b) << 1)

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
INF = float('inf')
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        if amount < 1 or len(coins) < 1:
            return 0
        
        min_coins = [0] * (amount + 1)
        
        for cur_amount in range(1, amount+1):
            min_coin_at_cur_amount = INF
            for coin in coins:
                if coin <= cur_amount and min_coins[cur_amount - coin] != INF:
                    min_coin_at_cur_amount = min(min_coin_at_cur_amount, min_coins[cur_amount - coin] + 1)
            min_coins[cur_amount] = min_coin_at_cur_amount
            
        return min_coins[amount] if min_coins[amount] != INF else -1

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
# https://leetcode.com/problems/climbing-stairs/
class Solution:
    def numDecodings(self, s: str) -> int:
        n = len(s)
        dp = [0] * (n+1)
        dp[0] = 1 # base case
        for i in range(1, n+1):
            c_index = i-1
            char = s[c_index]
            ways = 0
            if s[c_index] != '0': 
                ways += dp[i-1]
            
            if c_index - 1 >= 0:
                prev_char = s[c_index-1]
                if (prev_char == '1' and char <= '9') or (prev_char == '2' and char <= '6'):
                    ways += dp[i-2]
            dp[i] = ways
        
        return dp[n]

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
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        def build_graph(n, edges):
            graph = {}
            for edge in edges:
                q, p = edge[0], edge[1]
                if q in graph:
                    graph[q].append(p)
                else:
                    graph[q] = [p]
            return graph
        
        def found_cycle(start, graph, visited, finished):
            visited.add(start)
            if start in graph:
                for neighbor in graph[start]:
                    if neighbor not in visited:
                        if found_cycle(neighbor, graph, visited, finished):
                            return True
                    elif neighbor not in finished: # ancestor
                        return True
            finished.add(start)
            return False
                    
        graph = build_graph(numCourses, prerequisites)
        visited = set()
        finished = set()
        
        for i in range(numCourses):
            if i in visited:
                continue
            if found_cycle(i, graph, visited, finished):
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
class Solution:
    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        index = bisect.bisect_left(intervals, newInterval[0], key=lambda tup:tup[0])
        intervals.insert(index, newInterval)
        
        new_intervals = []
        for interval in intervals:
            if not new_intervals or interval[0] > new_intervals[-1][1]:
                new_intervals.append(interval)
            else:
                new_intervals[-1][1] = max(new_intervals[-1][1], interval[1])
    
        return new_intervals

# https://leetcode.com/problems/merge-intervals/
class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        intervals.sort(key=lambda interval:(interval[0], interval[1]))
        new_intervals = []
        for interval in intervals:
            if not new_intervals or interval[0] > new_intervals[-1][1]:
                new_intervals.append(interval)
            else:
                new_intervals[-1][1] = max(new_intervals[-1][1], interval[1])
        
        return new_intervals

# https://leetcode.com/problems/non-overlapping-intervals/
# Greedily to find the max number of intervals can be execute -> min removals
# Equal to the time scheduling problem to find the max number of tasks :) 
# https://en.wikipedia.org/wiki/Interval_scheduling
# Selecting the intervals that start earliest is not an optimal solution, because if the earliest interval happens to be very long, accepting it would make us reject many other shorter requests.
# Selecting the shortest intervals or selecting intervals with the fewest conflicts is also not optimal.
# [Earliest deadline first scheduling] The queue will be searched for the process closest to its deadline
class Solution:
    def eraseOverlapIntervals(self, intervals: List[List[int]]) -> int:
        intervals.sort(key=lambda interval:interval[1])
        print(intervals)
        ans = 0
        cur = None
        for interval in intervals:
            if cur is None or cur[1] <= interval[0]:
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
        while fast and fast.next:
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
        
        while tail.next:
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
