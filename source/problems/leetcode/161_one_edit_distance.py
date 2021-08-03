# https://leetcode.com/problems/one-edit-distance/

class Solution:
    def isOneEditDistance(self, s: str, t: str) -> bool:
        n = len(s)
        m = len(t)
        
        def rec(i, j, slot):
            if slot < 0:
                return False
            
            if i == n and j == m:
                return True if slot == 0 else False
            
            if i == n:
                return (m-j) == slot
            
            if j == m:
                return (n-i) == slot
            
            if s[i] == t[j]:
                return rec(i+1, j+1, slot)
            else:
                slot -= 1
                return rec(i+1, j, slot) or rec(i, j+1, slot) or rec(i+1, j+1, slot)
            
        return rec(0,0,1) # this sol actually works for K number of edits