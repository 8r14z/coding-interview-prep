# This is the IDE I mainly use to practice

import heapq

n = int(input())
arr = [int(num) for num in input().split()]
heap = []

s = 0
for num in arr:
    heapq.heappush(heap, num)
    s += num
    while s < 0:
        s-= heapq.heappop(heap)

print(len(heap))
