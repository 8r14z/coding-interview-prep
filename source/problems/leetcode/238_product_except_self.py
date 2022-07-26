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
        