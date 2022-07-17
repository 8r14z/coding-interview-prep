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
        