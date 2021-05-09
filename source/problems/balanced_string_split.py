# Input: s = "RLRRLLRLRL"
# Output: 4
# Explanation: s can be split into "RL", "RRLL", "RL", "RL", each substring contains same number of 'L' and 'R'.
# https://leetcode.com/problems/split-a-string-in-balanced-strings/submissions/

def balancedStringSplit(s):
    balanced  = 0
    res = 0
    for i in range(len(s)):
        if s[i] == 'L':
            balanced += 1
        else:
            balanced -= 1

        if balanced == 0:
            res += 1

    return res