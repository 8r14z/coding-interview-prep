# https://leetcode.com/problems/flip-string-to-monotone-increasing/
from functools import lru_cache
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
                if prev == '0': # 01 -> either keep this or flip 1 to 0
                    return min(dp(s[i], i+1), 1 + dp('0', i+1))
                else: # 10 -> flip 0 to 1
                    return 1 + dp('1', i+1)
                
        return dp('0', 0)


# Time: 2*n -> O(n) 
# Same space


class Solution:
    def minFlipsMonoIncr(self, s: str) -> int:
        flips = 0
        count = 0
        
        for c in s:
            if c == '1':
                count += 1
            else:
                flips = min(flips + 1, count)
            
        return flips

# O(n) & constant space
# https://leetcode.com/problems/flip-string-to-monotone-increasing/discuss/1394838/w-Hint-Python-O(n)-by-pattern-analysis