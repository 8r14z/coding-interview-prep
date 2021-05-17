# https://leetcode.com/problems/number-of-ways-to-rearrange-sticks-with-k-sticks-visible/
# DP by putting number in each case in the end and calculate what is the k number after that
#                              [1,2,3] n=3, k=2
#  put 1 in the end [2,3]       [1,3]         [1,2]
#                  n=2 k=2      n=2 k=2       n=2 k=1 (cuz 3-the biggest in the end is always visible)
#    stop here as n=k ^       same ^      [1] n=1 k=1   [2] n=1 k=0 
#                                   same pattern ^    stop ^
# n == k => there is only one way to arrage it ... in sorted order
# n < k => stop
class Solution:
    def rearrangeSticks(self, n: int, k: int) -> int:
        if n == k: return 1
        if n == 0 or k == 0 or n < k: return 0
        
        dp = [[0] * (k+1) for _ in range(n+1)]
        
        dp[0][0] = 1
        
        for i in range(1, n+1):
            for j in range(1, k+1):
                if i == j:
                    dp[i][j] = 1
                elif i < j:
                    dp[i][j] = 0
                else:
                    dp[i][j] = dp[i-1][j-1] + (i-1)*dp[i-1][j]
        
        return dp[n][k]
        