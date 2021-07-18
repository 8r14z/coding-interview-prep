# https://leetcode.com/problems/number-of-islands/

class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        n = len(grid)
        m = len(grid[0])
        
        def dfs(i, j):
            if i >= n or i < 0:
                return
            if j >= m or j < 0:
                return
            
            if grid[i][j] == '0' or grid[i][j] == '#':
                return
            
            grid[i][j] = '#'
            
            dfs(i+1,j)
            dfs(i-1,j)
            dfs(i,j+1)
            dfs(i,j-1)
        
        count = 0
        for i in range(n):
            for j in range(m):
                if grid[i][j] == '1':
                    dfs(i,j)
                    count += 1
        
        return count