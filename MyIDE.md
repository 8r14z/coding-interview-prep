from functools import cache
@cache
def minchange(A, N):
    if N == 0: return 0
    if N == 1: return 1

    arr = [minchange(A, N-c)+1 for c in A]
    return min(arr)


print(minchange([1,2,5], 10))

