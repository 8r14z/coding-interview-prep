def knapsack(iv, iw, w):
    n = len(iv)
    if n < 1: return 0

    dp = [[0] * (w+1) for _ in range(n+1)]

    for i in range(1,n+1):
        for j in range(1,w+1):
            dp[i][j] = max(
                dp[i-1][j], # excluding iw[j]
                iv[i-1] + (dp[i-1][j-iw[i-1]] if j > iw[i-1] else 0)
            )

    return dp[n][w]

print(knapsack([60, 100, 120], [10, 20, 30], 50))


def knapsack_repeated_items(W, items):
    dp = [0] * (W+1)

    for w in range(W+1):
        max_val = 0
        for iw, iv in items:
            if iw <= w:
                max_val = max(max_val, dp[w-iw] + iv)
        dp[w] = max_val
    
    return dp[W]

from functools import cache
def knapsack_repeated_recursion(W, items):
    @cache 
    def dp(w):
        if W == 0: return 0
        max_val = 0
        for iw, iv in items:
            if iw <= w:
                max_val = max(max_val, dp(w-iw) + iv)
        return max_val
    
    return dp(W)

print(knapsack_repeated_items(8, [(2,3), (3,5), (4,7), (5,9), (7,13)]))
print(knapsack_repeated_recursion(8, [(2,3), (3,5), (4,7), (5,9), (7,13)]))