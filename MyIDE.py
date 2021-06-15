# def solution(S, C):
#     n = len(S)
#     if n < 2: return 0
#     if len(C) != n: return 0

#     answer = 0
#     cur_sum = 0

#     for i in range(n-1):
#         if S[i] == S[i+1]:
#             cur_sum += C[i]
#         else:
#             answer += cur_sum
#             cur_sum = 0
    
#     answer += cur_sum

#     return answer


# print(solution('aabbcccc', [1, 2, 1, 2, 1, 2,3,4]))


# def solution(S, C):
#     n = len(S)
#     if n < 2: return 0
#     if len(C) != n: return 0

#     answer = 0
#     cur_sum = 0
#     cur_max = 0

#     tmp_s = S + '\n'

#     for i in range(n):
#         cur_max = max(cur_max, C[i])
#         cur_sum += C[i]    

#         if tmp_s[i] != tmp_s[i+1]:
#             answer += (cur_sum - cur_max)
#             cur_sum = 0
#             cur_max = 0

#     return answer


# print(solution('aabbcccc', [1, 2, 1, 2, 4, 3,2,1]))

# import collections

# n = int(input())
# columns = [cl for cl in input().split()]
# Row = collections.namedtuple('Row', ','.join(columns))

# sum = 0
# for _ in range(n):
#     cl = input().split()
#     row=Row(cl[0], cl[1], cl[2], cl[3])
#     sum+= int(row.MARKS)

# print('{:.2f}'.format(sum/n))


# '{:.3f}'.format(math.sqrt(5))


import math
import collections

def bfs(start, edges):    

    visited = set()
    visited.add(start)

    processed_edges = set()
    queue = collections.deque()
    queue.append(start)

    while queue:
        v = queue.popleft()
        for edge in edges:
            if edge in processed_edges: continue
            p,q = [int(e) for e in edge.split()]
            if v == p or v == q:
                next = q if v == p else p
                if next not in visited:
                    queue.append(next)
                    visited.add(next)
                    processed_edges.add(edge)
            
    return visited, processed_edges

def connectedSum(n, edges):
    processing_vertices = set(range(1,n+1))
    processing_edge = set(edges)
    sum = 0

    while processing_vertices:
        start = processing_vertices.pop()
        visited_vertices, visited_edges = bfs(start, processing_edge)    
        processing_vertices -= visited_vertices
        processing_edge -= visited_edges
        sum+= math.ceil(math.sqrt(len(visited_vertices)))    
        
    return sum
# O(n*m)

def solve(n, edges):
    def bfs(e, edges):
        vertices = [int(v) for v in e.split()]
        queue = collections.deque(vertices)
        visited = set(vertices)
        processed_edges = set()

        while queue:
            v = queue.popleft()
            for edge in edges:
                if edge in processed_edges: continue
                p,q = [int(v) for v in edge.split()]
                if v == p or v == q:
                    next = p if v == q else q
                    queue.append(next)
                    visited.add(next)
                    processed_edges.add(edge)
        
        return visited, processed_edges

    visited_vertices = set()
    processing_edges = set(edges)
    sum = 0

    while processing_edges:
        edge = processing_edges.pop()
        v,e = bfs(edge, processing_edges)
        processing_edges.difference_update(e)
        visited_vertices.update(v)
        sum += math.ceil(math.sqrt(len(v)))

    return n-len(visited_vertices) + sum
# O(m^2)
print(solve(8, ['8 1','5 8', '7 3', '8 6']))
print(solve(10, ['1 2','1 3', '2 4', '3 5', ' 8 7']))

Store = collections.namedtuple('Store', 'index capacity')
def get_store_capacity(s):
    length = len(s)
    is_open = False
    
    sum_array = []
    running_sum = 0
    count = 0
    
    for i in range(length):
        if s[i] == '|':
            running_sum += count
            sum_array.append(Store(i, running_sum))
            if not is_open: # open
                is_open = True
            else: # close
                count = 0
        else:
            if is_open: count += 1
            
    return sum_array

def closest_store(index, pairs, is_start):
    low = 0
    high = len(pairs)-1
    
    ans = float('inf') if is_start else float('-inf')

    while low <= high:
        mid = (low+high)//2
        start_index = pairs[mid][0]
        if start_index < index:
            if not is_start: ans = max(ans, mid)
            low = mid+1
        elif start_index > index:
            if is_start: ans = min(ans,mid)
            high = mid-1
        else: return pairs[mid]
    return pairs[ans]

def numberOfItems(s, startIndices, endIndices):
    ans = []
    n = len(startIndices)

    stores = get_store_capacity(s)
    
    for i in range(n):
        start = startIndices[i]-1
        end = endIndices[i]-1

        closest_start_store = closest_store(start, stores, True)
        closest_end_store = closest_store(end, stores, False)
        
        count = 0
        if closest_end_store.index >= closest_start_store.index:
            count = closest_end_store.capacity - closest_start_store.capacity
        
        ans.append(count)

    return ans

                    # 12345678
assert(numberOfItems('*|**|*|*', [1,1,2,2,1,3], [8,3,6,7,6,4]) == [3,0,2,3,2,0])
# Time: O(n + mlogn)
# Space: O(n+m)