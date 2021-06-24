def knapsack(iv, iw, w):
    n = len(iv)
    if n < 1: return 0

    dp = [[0] * (w+1) for _ in range(n+1)]

    for i in range(1,n+1):
        for j in range(1,w+1):
            dp[i][j] = max(
                dp[i-1][j],
                iv[i-1] + (dp[i-1][j-iw[i-1]] if j > iw[i-1] else 0)
            )

    return dp[n][w]

print(knapsack([60, 100, 120], [10, 20, 30], 50))


    