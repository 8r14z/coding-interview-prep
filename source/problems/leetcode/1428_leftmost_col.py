# https://leetcode.com/problems/leftmost-column-with-at-least-a-one/

INF = float('inf')
class Solution:
    def leftMostColumnWithOne(self, binaryMatrix: 'BinaryMatrix') -> int:
        rows, columns = binaryMatrix.dimensions()
        matrix = binaryMatrix
        
        result = INF
        i = 0
        j = columns-1
        
        while i < rows and j >= 0:
            if matrix.get(i, j):
                result = min(result, j)
                j -= 1
            else:
                i += 1
        
        return -1 if result == INF else result
                   