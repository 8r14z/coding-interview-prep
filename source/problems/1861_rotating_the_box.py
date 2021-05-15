# https://leetcode.com/problems/rotating-the-box/
class Solution:
    # O(n*m)
    # def rotateTheBox(self, box: List[List[str]]) -> List[List[str]]:
    #     n = len(box)
    #     m = len(box[0])
    #     output = [['.'] * n for _ in range(m)]
        
    #     for i in range(n):
    #         stoneCount = []
    #         count = 0
    #         for j in range(m):
    #             if box[i][j] == '*':
    #                 stoneCount.append((j, count))
    #                 count = 0
    #             elif box[i][j] == '#':
    #                 count += 1
                
    #         if count > 0:
    #             stoneCount.append((m, count))
            
    #         for countEntry in stoneCount:
    #             obstacleIndex = countEntry[0] 
    #             numOfStones = countEntry[1]
    #             for j in range(obstacleIndex-numOfStones, obstacleIndex):
    #                 output[j][n-1-i] = '#'
    #             if obstacleIndex < m:
    #                 output[obstacleIndex][n-1-i] = '*'
                
    #     return output
                
    # O(n*m) with 2 pointers
    def rotateTheBox(self, box: List[List[str]]) -> List[List[str]]:
        n = len(box)
        m = len(box[0])
        
        for i in range(n):
            lastEmpty = m-1
            for j in range(m-1, -1, -1):
                if box[i][j] == '*':
                    lastEmpty = j-1
                elif box[i][j] == '#':
                    box[i][j], box[i][lastEmpty] = box[i][lastEmpty], box[i][j]
                    lastEmpty -= 1
        
        output = [['.'] * n for _ in range(m)]
        for i in range(m):
            for j in range(n):
                output[i][j] = box[n-1-j][i]

        return output
            
                