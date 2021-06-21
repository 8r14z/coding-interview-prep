# Time O(n) Space: O(n)
def number_of_way_to_reach_stair_case(N):
    if N < 0:
        return 0
    if N <= 1: 
        return 1

    dp = [0] * (N+1)
    dp[0] = 1
    dp[1] = 1
    
    for i in range(2, N+1):
        dp[i] = dp[i-1] + dp[i-2]

    return dp[N]

# Time O(n) Space: O(1)
def number_of_way_to_reach_stair_case_constant_space(N):
    if N < 0:
        return 0
    if N <= 1: 
        return 1

    a = 1
    b = 1
    for i in range(2, N+1):
        c = a + b 
        a = b
        b = c

    return b

from functools import cache
# Time O(n) Space: O(n)
@cache
def number_of_way_to_reach_stair_case_dp(N):
    if N < 0:
        return 0
    if N <= 1: 
        return 1

    return number_of_way_to_reach_stair_case_dp(N-1) + number_of_way_to_reach_stair_case_dp(N-2)

print(number_of_way_to_reach_stair_case(6)) # 13


# 1 or M steps

from collections import deque
def stair_case_constant_space(N):
    if N < 0:
        return 0
    if N < 5:
        return 1

    queue = deque()
    for _ in range(5):
        queue.append(1)

    for _ in range(5, N+1):
        c = queue[0] + queue[-1]
        queue.popleft()
        queue.append(c)

    return queue[-1]

def stair_case(N):
    if N < 0:
        return 0
    if N < 5:
        return 1

    dp = [0] * (N+1)
    dp[0] = dp[1] = 1

    for i in range(2, N+1):
        dp[i] = dp[i-1] + (0 if i < 5 else dp[i-5])

    return dp[N]

print(stair_case(6))