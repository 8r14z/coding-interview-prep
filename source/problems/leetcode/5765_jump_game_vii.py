# https://leetcode.com/problems/jump-game-vii/
class Solution:
    def canReach(self, s: str, minJump: int, maxJump: int) -> bool:
        n = len(s)
        if n < 2: return False
        if s[0] == 1 or s[-1] == '1': return False
        
        dp = [False] * n
        dp[-1] = True
        
        diff = maxJump - minJump
        start = end = n-1
        valid_count = 1
        
        for i in range(n-1-minJump, -1, -1):
            if s[i] == '0' and valid_count > 0:
                dp[i] = True
                
            start -= 1
            if dp[start]:
                valid_count += 1
            if end-start > diff:
                if dp[end]:
                    valid_count -= 1
                end -= 1
            
        return dp[0]
        