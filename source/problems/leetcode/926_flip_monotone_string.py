# https://leetcode.com/problems/flip-string-to-monotone-increasing/

class Solution:
    def minFlipsMonoIncr(self, s: str) -> int:
        n = len(s)
        @lru_cache(None)
        def dp(prev, i) -> int:
            if i == n:
                return 0
            
            if s[i] == prev:
                return dp(s[i], i+1)
            else:
                if prev == '0':
                    return min(dp(s[i], i+1), 1 + dp('0', i+1))
                else:
                    return 1 + dp('1', i+1)
                
        return dp('0', 0)


# Time: 2*n -> O(n) 
# Same space