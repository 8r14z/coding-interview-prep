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
    def search(subsequence, val) -> int:
        low = 0
        high = len(subsequence) - 1

        index = high
        while low <= high:
            mid = (low + high) // 2
            if (subsequence[mid] >= val):
                index = min(index, mid)
                high = mid - 1
            else:
                low = mid + 1

        return index

    def lengthOfLIS(self, nums: List[int]) -> int:
        n = len(nums)
        if not n:
            return 0
        if n == 1:
            return 1
        
        subsequence = []
        subsequence.append(nums[0])
        
        for i in range(1, n):
            index = self.search(subsequence, nums[i])
            if (index < len(subsequence)):
                subsequence[index] = nums[i]
            else:
                subsequence.append(nums[i])
        
        
        return len(subsequence)