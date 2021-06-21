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


# for the problem with steps are either 1 or X, X > 2. we can apply DP with X == 2, it's fibonacci