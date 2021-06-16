
def max_value(W, items):
    dp = [0] * (W+1)

    for w in range(W+1):
        max_val = 0
        for iw, iv in items:
            if iw <= w:
                max_val = max(max_val, dp[w-iw] + iv)
        dp[w] = max_val
    
    return dp[W]


print(max_value(8, [(2,3), (3,5), (4,7), (5,9), (7,13)]))
