def longest_common_subsequence(str1, str2):
    if not str1 or not str2:
        return 0

    newStr1 = str1[0:len(str1) - 1]
    newStr2 = str2[0:len(str2) - 1]

    if str1[-1] == str2[-1]:
        return 1 + longest_common_subsequence(newStr1, newStr2)
    else:
        return max(longest_common_subsequence(newStr1, str2), longest_common_subsequence(str1, newStr2))

# DP approach
def longest_common_subsequence_dp(str1, str2):
    res = 0
    dp = [[0] * (len(str2) + 1) for _ in range(len(str1) + 1)]

    for i in range(1, len(str1)+1):
        for j in range(1, len(str2)+1):
            if str1[i-1] == str2[j-1]:
                dp[i][j] = 1 + dp[i-1][j-1]
            else:
                dp[i][j] = max(dp[i][j-1], dp[i-1][j])
            res = max(res, dp[i][j])

    return res

print(longest_common_subsequence('abdjsh', 'ads')) # a d s
