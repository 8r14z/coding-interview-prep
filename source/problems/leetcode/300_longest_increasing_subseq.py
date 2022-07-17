# https://leetcode.com/problems/longest-increasing-subsequence/

# Memorization
INF = float('inf')
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        n = len(nums)
        if n < 1:
            return 0
        
        dp = [1] * n
        answer = -INF
        for i in reversed(range(n)):
            for j in range(i+1, n):
                if nums[j] > nums[i]:
                    dp[i] = max(dp[i], 1 + dp[j])
            answer = max(answer, dp[i])
             
        return answer

# Binary search 
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
        