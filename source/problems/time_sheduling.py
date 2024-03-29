# interval (start, end)
# O(nlogn) for sorting
# greedly find the earliest finish time :) 
# https://www.youtube.com/watch?v=R6Skj4bT1HE
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
# O(nlogn)
# https://www.geeksforgeeks.org/weighted-job-scheduling-log-n-time/
def weight_time_scheduling(intervals):
    # sort by start time
    sorted_intervals = sorted(intervals, key=lambda tup: tup[0])
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
        dp_compatible = 0 if compatible_index == -1 else dp[compatible_index]
        w = sorted_intervals[i][2]
        dp[i] = max(dp[i+1], w + dp_compatible)
        res = max(res, dp[i])

    return res

print(weight_time_scheduling([(1,4,1), (4,6,1), (6,8,1), (3,5,2), (5,7,2)]))
        