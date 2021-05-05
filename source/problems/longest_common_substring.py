def longest_common_substring_dp(str1, str2):
    res = 0
    dp = [[0] * len(str2) for _ in range(len(str1))]

    for i in range(len(str1)):
        for j in range(len(str2)):
            if str1[i] == str2[j]:
                dp[i][j] = 1 + (dp[i-1][j-1] if i > 0 and j > 0 else 0)
            else:
                dp[i][j] = 0
            res = max(res, dp[i][j])
    print(dp)
    return res

# print(longest_common_substring('fish', 'hish'))
# print(longest_common_substring('fisdishe', 'hishefcd'))