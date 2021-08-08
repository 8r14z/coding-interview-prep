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