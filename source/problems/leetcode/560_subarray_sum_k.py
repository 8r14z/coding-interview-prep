# https://leetcode.com/problems/subarray-sum-equals-k/
class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        n = len(nums)
        count = 0
        prefix_sum = 0 # cumulative sum at i sum(0,i)
        seen_sum = defaultdict(int)
        
        for num in nums:
            prefix_sum += num
            if prefix_sum == k:
                count += 1
                
            if prefix_sum-k in seen_sum: # seen cumulative sum of previous indecies
                count += seen_sum[prefix_sum-k]
            
            seen_sum[prefix_sum] += 1
        
        return count