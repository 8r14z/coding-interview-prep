def max_sum_subarray(arr, k):
    max_sum = -1
    start = 0
    cur_sum = 0

    for end, val in enumerate(arr):
        cur_sum += val
        if end - start + 1 == k:
            max_sum = max(max_sum, cur_sum)
            cur_sum -= arr[start]
            start += 1

    return max_sum

print(max_sum_subarray([2, 3, 4, 1, 5], 3))
