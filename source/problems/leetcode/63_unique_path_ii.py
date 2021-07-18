# https://leetcode.com/problems/unique-paths-ii/

class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        r = len(obstacleGrid)
        if r < 1: return 0
        c = len(obstacleGrid[0])
        if c < 1: return 0
        
        if obstacleGrid[0][0] == 1:
            return 0
        
        dp = [[0]*(c+1) for _ in range(r+1)]
        
        for i in range(1, r+1):
            for j in range(1, c+1):
                if i == 1 and j == 1:
                    dp[i][j] = 1
                else:
                    if not obstacleGrid[i-1][j-1]:
                        dp[i][j] = dp[i-1][j] + dp[i][j-1]

        return dp[r][c]