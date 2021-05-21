# https://leetcode.com/problems/minimum-number-of-swaps-to-make-the-binary-string-alternating/
class Solution:
    def solve(self, s: str, even_mode: str) -> int:
        wrong_even = 0
        wrong_odd = 0
        
        for i in range(len(s)):
            if i % 2 == 0 and s[i] != even_mode:
                wrong_even += 1
            elif i % 2 == 1 and s[i] == even_mode:
                wrong_odd += 1
        if wrong_even != wrong_odd:
            return float('inf')
        else:
            return wrong_even
        
    def minSwaps(self, s: str) -> int:
        res = min(self.solve(s, '0'), self.solve(s, '1'))
        return -1 if res == float('inf') else res
                    
                