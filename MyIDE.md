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


# interval (start, end, weight)
def weight_time_scheduling(intervals):
    sorted_intervals = sorted(intervals, key=lambda tup: tup[1])
    n = len(intervals)
    dp = [0] * n
    dp[n-1] = sorted_intervals[n-1][2]
    
    def find_compatible(i, intervals) -> int: 
        n = len(intervals)
        left = i + 1
        right = n - 1
        f = intervals[i][1]
        res = n

        while left <= right:
            mid = (left+right)//2
            s = intervals[mid][0] 
            if s > f:
                res = min(res, s)
                right = mid - 1
            elif s < f:
                left = mid + 1
            else:
                return mid
        return -1 if res == n else res

    res = float('-inf')
    for i in reversed(range(n-1)):
        compatible_index = find_compatible(i, sorted_intervals)
        w = sorted_intervals[i][2]
        dp[i] = w if compatible_index == -1 else w + sorted_intervals[compatible_index][2]
        res = max(res, dp[i])
    return max

print(weight_time_scheduling([(1,4,1), (4,6,1), (6,8,1), (3,5,2), (5,7,2)]))
        