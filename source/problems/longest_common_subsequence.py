def longest_common_subsequence(str1, str2):
    if not str1 or not str2:
        return 0

    newStr1 = str1[0:len(str1) - 1]
    newStr2 = str2[0:len(str2) - 1]

    if str1[-1] == str2[-1]:
        return 1 + longest_common_subsequence(newStr1, newStr2)
    else:
        return max(longest_common_subsequence(newStr1, str2), longest_common_subsequence(str1, newStr2))

print(longest_common_subsequence('abdjsh', 'ads')) # a d s
