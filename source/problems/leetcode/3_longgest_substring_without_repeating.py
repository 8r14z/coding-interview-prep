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

    # O(n) - O(1)
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        
        n = len(nums)
        if (n <= 1):
            return nums
        
        ans = [0] * n
        
        prefix = nums[0]
        for i in range(1, n):
            ans[i] = prefix
            prefix *= nums[i]
        
        postfix = nums[-1]
        for i in reversed(range(n-1)):
            ans[i] = postfix if i == 0 else ans[i] * postfix
            postfix *= nums[i]
        
        return ans