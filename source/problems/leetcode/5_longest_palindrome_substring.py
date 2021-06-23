# [https://leetcode.com/problems/longest-palindromic-substring/]

def longestPalindromeDp(s):
    rev = s[::-1]
    n = len(s)
    dp = [[0] * n for _ in range(n)]
    max_len = 0
    max_end = 0

    for i in range(n):
        for j in range(n):
            if s[i] == rev[j]:
                dp[i][j] = 1 + (dp[i-1][j-1] if i > 0 and j > 0 else 0)
            else:
                dp[i][j] = 0

            if dp[i][j] > max_len and j - dp[i][j] + 1 == len(s) - 1 - i:
                # start index of duplicated substring [x] should be same as index  on reversed string in revered order
                # index x = 2 -> y = len(s) - 1 - x
                max_len = dp[i][j]
                max_end = i

    return s[max_end - max_len + 1: max_end+1]


def longest_palindrome_substring(s):
    n = len(s)
    if n < 1: return 0
    if n == 1: return 1

    dp = [[False] * n for _ in range(n)]

    res = 1
    # palindrome len = 1
    for i in range(n):
        dp[i][i] = 1
    
    # palindrome len = 2
    for i in range(n-1):
        dp[i][i+1] = 2 if s[i] == s[i+1] else 0
        res = max(res, dp[i][i+1])

    # len >= 3
    for l in range(3, n+1):
        for i in range(n-l+1):
            j = i + l - 1
            if s[i] == s[j] and dp[i+1][j-1]:
                dp[i][j] = True
                res = max(res, l)

    return res

print('longest_palindrome_dp')
print(longest_palindrome_substring('forgeeksskeegfor'))

# O(n^2) with less constant factor
class Solution:
    def longestPalindrome(self, s: str) -> str:
        if s == '': 
            return s
            
        start, end = 0, 0
        for i in range(len(s)):
            len1 = self.expandFromCenter(s, i, i)
            len2 = self.expandFromCenter(s, i, i+1)
            l = max(len1, len2)
            if l > end - start + 1:
                start = i - (l - 1)//2
                end = i + l//2

        return s[start:end+1]

    def expandFromCenter(self, s, l, r):
        while l >= 0 and r < len(s) and s[l] == s[r]:
            l -= 1
            r += 1
        return r - l - 1 
        

print(longestPalindromeDp("babcadcbab"))
