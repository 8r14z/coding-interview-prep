# https://leetcode.com/problems/unique-paths/

class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        if m < 1 or n < 1:
            return 0
        
        dp = [[0] * (n+1) for _ in range(m+1)]
        
        for i in range(1,m+1):
            for j in range(1, n+1):
                if i == 1 and j == 1:
                    dp[i][j] = 1
                else:
                    dp[i][j] = dp[i-1][j] + dp[i][j-1]
        
        return dp[m][n]
    

    