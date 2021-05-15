# https://leetcode.com/problems/minimum-number-of-operations-to-move-all-balls-to-each-box/

# O(n)
def minOperations(boxes) -> [int]:
    n = len(boxes)
    ballCount = 0
    res1 = [0 for _ in range(n)]
    res2 = [0 for _ in range(n)]

    for i in range(n):
        if i > 0: res1[i] = res1[i-1] + ballCount
        if (boxes[i] == '1'):
            ballCount += 1
    
    ballCount = 0
    for i in range(n-1, -1, -1):
        if i < n-1: res2[i] = res2[i+1] + ballCount
        if (boxes[i] == '1'):
            ballCount += 1

    res = [0 for _ in range(n)]
    for i in range(n):
        res[i] = res1[i] + res2[i]
    return res
        
print(minOperations("001011"))
# [11,8,5,4,3,4]


# 0 0 1 0 1 1
# 