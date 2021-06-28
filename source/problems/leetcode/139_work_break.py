# https://leetcode.com/problems/word-break/

dictionary = set([
    'apple',
    'pear',
    'pie'
])

from functools import cache
def work_break_td(string, dictionary):
    n = len(string)
    @cache
    def dp(start, end):
        word = string[start:end+1]
        if start >= n or end >= n:
            return True if word in dictionary else False

        if word in dictionary:
            return dp(start, end+1) or dp(end+1, end+1)
        else:
            return dp(start, end+1)

    return dp(0,0)

def work_break_bu(s, dictionary):
    n = len(s)
    dp = [False] * n

    for i in reversed(range(n)):
        for j in range(i, n):
            word = s[i:j+1]
            print(word)
            if word in dictionary:
                if j == n-1 or dp[j+1]:
                    dp[i] = True
                    break
    return dp[0]


print(work_break_td('applepie', dictionary))
print(work_break_bu('applepie', dictionary))