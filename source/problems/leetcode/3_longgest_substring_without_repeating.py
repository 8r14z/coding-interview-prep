# https://leetcode.com/problems/longest-substring-without-repeating-characters/

class Solution:
    # "abcabcbb"
    def lengthOfLongestSubstring(self, s: str) -> int:
        n = len(s)
        if n < 1: return 0

        res = float('-inf')
        existed = {}
        start = 0
        for end, c in enumerate(s):
            if c in existed and existed[c] >= start:
                index = existed[c]
                start = index + 1

            existed[c] = end

            res = max(res, end-start+1)

        return res