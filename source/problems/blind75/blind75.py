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
        cur_max = 0
        res = -float('inf')
        for i in range(n):
            cur_max = max(cur_max + nums[i], nums[i])
            res = max(res, cur_max)
            
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
