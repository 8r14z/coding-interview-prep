# https://leetcode.com/problems/maximum-subarray/
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        n = len(nums)
        
        local_max = nums[0]
        result = local_max
        
        for i in range(1, n):
            local_max = max(local_max + nums[i], nums[i])
            result = max(result, local_max)
        
        return result