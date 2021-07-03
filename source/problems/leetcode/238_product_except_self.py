# https://leetcode.com/problems/product-of-array-except-self/

class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:        
        n = len(nums)

        if n < 2:
            return []
        
        left = [0] * n
        left[0] = nums[0]
        
        right = [0] * n
        right[-1] = nums[-1]
        
        for i in range(1, n):
            left[i] = nums[i] * left[i-1]
        
        for i in reversed(range(n-1)):
            right[i] = nums[i] * right[i+1]
            
        res = []
        for i in range(n):
            if i == 0:
                res.append(right[i+1])
            elif i == n-1:
                res.append(left[i-1])
            else:
                res.append(left[i-1] * right[i+1])
        
        return res