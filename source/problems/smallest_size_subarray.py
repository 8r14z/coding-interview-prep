def smallest_size_subarray(arr, k):
    min_size = -1
    start = 0
    cur_sum = 0
    for end, val in enumerate(arr):
        cur_sum += val
        while cur_sum >= k:
            min_size = end - start + 1 if min_size == - \
                1 else min(min_size, end - start + 1)
            cur_sum -= arr[start]
            start += 1

    return min_size

print(smallest_size_subarray([2, 4, 2, 5, 1], 7))
