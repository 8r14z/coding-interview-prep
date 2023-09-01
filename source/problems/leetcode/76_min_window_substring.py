# https://leetcode.com/problems/minimum-window-substring/description/
from collections import defaultdict

class Solution:
    def minWindow(self, s: str, t: str) -> str:
        tFrequency = defaultdict(int)
        for c in t:
            tFrequency[c] += 1
    
        charCounter = [0] * 128

        def contains():
            for char in tFrequency:
                if tFrequency[char] > charCounter[ord(char)]:
                    return False
            return True

        rStart = rEnd = -1
        start = 0
        for end in range(len(s)):
            char = s[end]
            charCounter[ord(char)] += 1
            
            while contains():
                if rStart == -1 or rEnd-rStart+1 >= end-start+1:
                    rStart = start
                    rEnd = end
                
                charCounter[ord(s[start])] -= 1
                start += 1


        return s[rStart:rEnd+1]