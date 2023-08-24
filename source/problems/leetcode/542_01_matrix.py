# https://leetcode.com/problems/01-matrix/description/
# O(m * n)

class Solution:
    def updateMatrix(self, mat: List[List[int]]) -> List[List[int]]:
        if len(mat) == 0:
            return []

        m = len(mat)
        n = len(mat[0])

        distanceMat = [[0] * n for _ in range(m)]
        queue = deque()
        visited = set()
        for i in range(m):
            for j in range(n):
                if mat[i][j] == 0:
                    v = (i,j)
                    queue.append(v)
                    visited.add(v)

        dist = 0
        while queue:
            count = len(queue)
            for _ in range(count):
                i,j = queue.popleft()
                
                neighbors = []
                if i > 0: 
                    neighbors.append((i-1, j))
                if i < m-1:
                    neighbors.append((i+1, j))
                if j > 0:
                    neighbors.append((i, j-1))
                if j < n-1:
                    neighbors.append((i, j+1))

                for neighbor in neighbors:
                    if neighbor not in visited:
                        queue.append(neighbor)
                        visited.add(neighbor)
                        ni,nj = neighbor
                        distanceMat[ni][nj] = dist + 1

            dist += 1

        return distanceMat