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

from collections import deque
# 1 or 5 steps
def stair_case1(N):
    if N < 0:
        return 0
    if N < 5:
        return 1

    queue = deque()
    for _ in range(5):
        queue.append(1)

    for _ in range(5, N+1):
        c = queue[0] + queue[-1]
        queue.popleft()
        queue.append(c)

    return queue[-1]

def stair_case(N):
    if N < 0:
        return 0
    if N < 5:
        return 1

    dp = [0] * (N+1)
    dp[0] = dp[1] = 1

    for i in range(2, N+1):
        dp[i] = dp[i-1] + (0 if i < 5 else dp[i-5])

    return dp[N]


print(stair_case1(6))
print(stair_case(6))

print(stair_case1(7))
print(stair_case(7))

print(stair_case1(15))
print(stair_case(15))

from collections import deque
import math
def solve_generate_graph(n, edges): 
    graph = {}

    for edge in edges:
        p,q = [int(v) for v in edge.split()] 
        if p in graph:
            graph[p].append(q)
        else:
            graph[p] = [q]
        if q in graph:
            graph[q].append(p)
        else:
            graph[q] = [p]
    
    def bfs(start, graph):
        queue = deque([start])
        visited = set([start])

        while queue:
            current_vertex = queue.popleft()
            neighbors = graph[current_vertex]
            
            for vertex in neighbors:
                if vertex not in visited: 
                    visted.add(vertex)
                    queue.append(vertex)
        return visited

    processing_vertices = set(range(1, n+1))
    sum = 0
    while processing_vertices:
        start = processing_vertices.pop()
        visted_vertices = bfs(start, graph)
        processing_vertices -= visited_vertices
        sum += math.ceil(math.sqrt(len(visited_vertices)))

    return sum + (n-len(graph))

print(solve_generate_graph(8, ['8 1','5 8', '7 3', '8 6']))
print(solve_generate_graph(10, ['1 2','1 3', '2 4', '3 5', ' 8 7']))