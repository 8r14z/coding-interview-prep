# https://leetcode.com/problems/word-break-ii/

class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> List[str]:
        n = len(s)
        dictionary = set(wordDict)
        dp = [[] for _ in range(n)]
        
        for i in reversed(range(n)):
            for j in range(i, n):
                word = s[i:j+1]
                
                if word in dictionary:
                    if j == n-1 or len(dp[j+1]) != 0:
                        dp[i].append(word)
        
    
        if len(dp[0]) == 0:
            return []
        
        answer = []
        def dfs(i, tmp):
            if i >= n: 
                answer.append(' '.join(tmp))
                return
            
            for word in dp[i]:
                tmp.append(word)
                dfs(i+len(word), tmp)
                tmp.pop()
            
        dfs(0, [])
        return answer