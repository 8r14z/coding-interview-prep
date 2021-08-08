# https://leetcode.com/problems/palindrome-partitioning-ii/

class Solution:
    def minCut(self, s: str) -> int:
        n = len(s)
        dp = [1] * n
        palindrome = [[False] * n for _ in range(n)]
        
        # Step 1: Build the palindrome map
        # to check whether substring from start - end is palindrome or not
        for start in reversed(range(n)):
            for end in range(start,n):
                if s[start] == s[end] and (end-start <= 2 or palindrome[start+1][end-1]):
                    palindrome[start][end] = True
        
        # Step 2: intuition is to find min number of palindrome can be made at index i
        # if substring i -> j is a palindrome => substring starts at j+1 also palindrome
        # => find all pairs start with i and get the min number of palindroms
        # result is at i == 0
        # The problem asks about min cut => only cut between palindromes. i.e: have n palindromes => cuts = n-1
        for i in reversed(range(n)):
            res = float('inf')
            for j in range(i,n):
                if palindrome[i][j]:
                    if j < n-1:
                        res = min(res, 1 + dp[j+1])
                    else:
                        res = 1
            dp[i] = res
            
        return dp[0]-1


class Solution:
    def minCut(self, s: str) -> int:
        def expand_from_center(s, start, end, dp):
            while start >= 0 and end < len(s) and s[start] == s[end]:
                cuts = 0 if start == 0 else (dp[start-1] + 1)
                dp[end] = min(dp[end], cuts)
                start -= 1
                end += 1
                
        n = len(s)
        dp = [0] * n
        
        for i in range(n):
            dp[i] = i
            
        for i in range(n):
            expand_from_center(s, i, i, dp)
            expand_from_center(s, i, i+1, dp)
            
        return dp[n-1]
                