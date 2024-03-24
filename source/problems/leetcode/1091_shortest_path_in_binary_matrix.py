# https://leetcode.com/problems/shortest-path-in-binary-matrix/

from collections import defaultdict
from collections import deque

class Solution:
    def shortestPathBinaryMatrix(self, grid: List[List[int]]) -> int:
        if grid[0][0] != 0:
            return -1
        m = len(grid)
        n = len(grid[-1])

        shortestPath = defaultdict(int)
        shortestPath[(0,0)] += 1
        queue = deque([(0,0)])

        while queue:
            i,j = queue.popleft()
            if i == m-1 and j == n-1:
                return shortestPath[(i,j)]
            
            next_directions = [[-1,0], [-1,1], [0,1], [1,1], [1,0], [1,-1], [0,-1],[-1,-1]]
            for dir in next_directions:
                next_i = dir[0] + i
                next_j = dir[1] + j
                if next_i >= 0 and next_i < m and next_j >= 0 and next_j < n and grid[next_i][next_j] == 0 and (next_i, next_j) not in shortestPath:
                    shortestPath[(next_i, next_j)] = shortestPath[(i,j)] + 1
                    queue.append((next_i, next_j))

        return -1