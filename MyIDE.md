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


def time_scheduling(intervals):
    sorted_intervals = sorted(intervals, key=lambda tup: tup[1])

    count = 0
    prev = None
    for interval in sorted_intervals:
        if prev is None or interval[0] >= prev[1]:
            prev = interval
            count+=1

    return count

print(time_scheduling([(1,3), (2,4), (1,4), (2,4), (4,6), (5,7), (6, 8), (8, 10), (9, 11)]))