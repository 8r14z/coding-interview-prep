# https://leetcode.com/problems/rotating-the-box/
class Solution:
    def rotateTheBox(self, box: List[List[str]]) -> List[List[str]]:
        n = len(box)
        m = len(box[0])
        output = [['.'] * n for _ in range(m)]
        
        for i in range(n):
            stoneCount = []
            count = 0
            for j in range(m):
                if box[i][j] == '*':
                    stoneCount.append((j, count))
                    count = 0
                elif box[i][j] == '#':
                    count += 1
                
            if count > 0:
                stoneCount.append((m, count))
            
            for countEntry in stoneCount:
                obstacleIndex = countEntry[0] 
                numOfStones = countEntry[1]
                for j in range(obstacleIndex-numOfStones, obstacleIndex):
                    output[j][n-1-i] = '#'
                if obstacleIndex < m:
                    output[obstacleIndex][n-1-i] = '*'
                
        return output
                
            
            
                