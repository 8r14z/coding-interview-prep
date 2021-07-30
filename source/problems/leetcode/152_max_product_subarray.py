
class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        n = len(nums)
        assert(n > 0)
        
        global_max = local_max = local_min = nums[0]
        
        for i in range(1, n):
            num = nums[i]
            cur_max = max(local_max * num, local_min * num, num)
            cur_min = min(local_max * num, local_min * num, num)
            
            local_max = cur_max
            local_min = cur_min
            
            global_max = max(global_max, local_max)
            
        return global_max