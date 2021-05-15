class Solution:
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
                
            
            
                