# https://leetcode.com/problems/longest-substring-with-at-most-k-distinct-characters/

class Solution:
    def lengthOfLongestSubstringKDistinct(self, s: str, k: int) -> int:        
        if k > len(s):
            return len(s)
        
        res = float('-inf')
        seen = defaultdict(int)
        start = 0
        
        for end, character in enumerate(s):
            seen[character] += 1
            if len(seen) > k:
                prev_char = s[start]
                if prev_char in seen:
                    seen[prev_char] -= 1
                    if seen[prev_char] == 0:
                        seen.pop(prev_char)
                start += 1
            res = max(res, end-start+1)
            
        return res