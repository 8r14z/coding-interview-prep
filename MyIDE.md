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

INF = float('inf')
def total_occurences(A, K):
    def binary_search(A, K, find_most_left):
        n = len(A) 
        left = 0 
        right = n - 1
        res = INF if find_most_left else -INF

        while left <= right:
            mid = (left+right) // 2
            if A[mid] < K:
                left = mid + 1 
            elif A[mid] > K:
                right = mid - 1
            else:
                if find_most_left:
                    res = min(res, mid)
                    right = mid - 1
                else:
                    res = max(res, mid)
                    left = mid + 1
                
        return res if res >= 0 and res < n else -1

    start_index = binary_search(A, K, True)
    if start_index == -1: return 0
    end_index = binary_search(A, K, False)
    return end_index - start_index + 1

print(total_occurences([1,2,2,2,7,10,12,20,99,100], 2))

# partition descreasing order
def partition(A, L, R):
    p_index = L # the last i that element < pivot
    pivot = A[R]

    for i in range(L, R):
        if A[i] >= pivot:
            A[p_index], A[i] = A[i], A[p_index]
            p_index += 1

    A[p_index], A[R] = A[R], A[p_index]
    return p_index

print(partition([1,3,3,3,3,3], 0, 5))