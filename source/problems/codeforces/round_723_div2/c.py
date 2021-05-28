# https://codeforces.com/contest/1526/problem/C2
import heapq

n = int(input())
arr = [int(num) for num in input().split()]
heap = []

s = 0
for num in arr:
    heapq.heappush(heap, num)
    s += num
    if s < 0:
        s-= heapq.heappop(heap)

print(len(heap))
