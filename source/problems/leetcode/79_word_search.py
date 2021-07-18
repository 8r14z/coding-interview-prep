# https://leetcode.com/problems/word-search/
# backtracking
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        l = len(word)
        n = len(board)
        m = len(board[0])
        VISITED = '*'
        
        def dfs(i, j, wi):
            if wi == l:
                return True
            
            if i >= n or i < 0:
                return False
            if j >= m or j < 0:
                return False
            if board[i][j] == VISITED:
                return False
            if board[i][j] != word[wi]:
                return False
            
            board[i][j] = VISITED
            found = dfs(i+1, j, wi+1) or dfs(i-1, j, wi+1) or dfs(i, j+1, wi+1) or dfs(i, j-1, wi+1)
            board[i][j] = word[wi]
            return found
        
        # Time O(n*m*3^l)
        # Space O(l)
        for i in range(n):
            for j in range(m):
                if dfs(i, j, 0):
                    return True
        
        return False