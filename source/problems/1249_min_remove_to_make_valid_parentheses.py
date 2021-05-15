# https://leetcode.com/problems/minimum-remove-to-make-valid-parentheses/

class Solution:
    def minRemoveToMakeValid(self, s: str) -> str:
        invalid_indices = set()
        parentheses = []
        
        for i, c in enumerate(s):
            if c == '(':
                parentheses.append(i)
            elif c == ')':
                if len(parentheses) == 0:
                    invalid_indices.add(i)
                else:
                    parentheses.pop()
                    
        res = ''
        for i in range(len(s)):
            if i not in parentheses and i not in invalid_indices:
                res += s[i]
                
        return res
                    
                