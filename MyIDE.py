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
from source.problems.stair_case import stair_case

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
print('first approach')
print(connectedSum(8, ['8 1','5 8', '7 3', '8 6']))
print(connectedSum(10, ['1 2','1 3', '2 4', '3 5', ' 8 7']))

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
print('second approach')
print(solve(8, ['8 1','5 8', '7 3', '8 6']))
print(solve(10, ['1 2','1 3', '2 4', '3 5', ' 8 7']))

from collections import deque
import math
def solve_generate_graph(n, edges): 
    def generate_graph(edges):
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

        return graph
    
    def bfs(start, graph):
        queue = deque()
        queue.append(start)
        visited = set([start])

        while queue:
            current_vertex = queue.popleft()
            if current_vertex not in graph:
                continue
            
            neighbors = graph[current_vertex]
            for vertex in neighbors:
                if vertex not in visited: 
                    visited.add(vertex)
                    queue.append(vertex)

        return visited

    graph = generate_graph(edges)
    processing_vertices = set(range(1, n+1))
    visited_count = 0
    sum = 0
    while processing_vertices:
        start = processing_vertices.pop()
        visited_vertices = bfs(start, graph)
        processing_vertices -= visited_vertices
        visited_count += len(visited_vertices)
        sum += math.ceil(math.sqrt(len(visited_vertices)))

    return sum + (n-visited_count)
# O(N+M) <- optimal solution 
# space O(N+M) <- n + 2m
print('third approach')
print(solve_generate_graph(8, ['8 1','5 8', '7 3', '8 6']))
print(solve_generate_graph(10, ['1 2','1 3', '2 4', '3 5', ' 8 7']))













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




class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
        
class Solution:
    def flatten(self, root: TreeNode) -> None:
        self.prev = None
        
        def preorder(node):
            if node is None: return
            print(node)
            if self.prev:
                self.prev.right = node
            self.prev = node
            print(node.val)
            
            preorder(node.left)
            preorder(node.right)
            
        preorder(root)
        return root


class File:
    def __init__(self):
        self.cache = []
        self.cur_index = 0

    def refresh_cache(self):
        if not self.cache or self.cur_index >= len(self.cache):
            chunk = self.read4k()
            if not chunk:
                return False

            self.cache = chunk
            self.cur_index = 0
        
        return True

    def read_line(self):
        is_succeeded = self.refresh_cache()

        if not is_succeeded: return ''

        start = self.cur_index
        end = self.find_next_end_of_line()

        if end != -1:
            self.cur_index = end + 1
            return self.cache[start:end+1]
        else:
            self.cur_index = len(self.cache)
            return self.cache[start:] + self.read_line()

    def find_next_end_of_line(self):
        start = self.cur_index
        end = -1
        for i in range(start, len(self.cache)):
            if self.cache[i] == '\n':
                end = i
                break
        return end

    def read_4k(self):
        pass


# 31
# 245

# 45
# 31

# 31
# 345

# 34531

# 32451

# 14
# ''

# 354
# 3514
# output = 3543514

# 354
# 354
# output = 3535414

# 31
# 245

def mergeNumbers(firstNumber, secondNumber):
    # Write your code here
    output = ''
    
    i = 0 # 1st number
    j = 0 # 2nd number
    n = len(firstNumber)
    m = len(secondNumber)
    
    while i < n or j < m:
        cur_i = i
        cur_j = j
 
        while cur_i < n and cur_j < m and firstNumber[cur_i] == secondNumber[cur_j]:
            cur_i += 1
            cur_j += 1
        
        if cur_i < n and cur_j < m:
            if firstNumber[cur_i] > secondNumber[cur_j]:
                output += firstNumber[i]
                i +=1
            else:
                output += secondNumber[j]
                j += 1
        elif cur_i < n:
            output += firstNumber[i]
            i += 1
        elif cur_j < m:
            output += secondNumber[j]
            j += 1 

    return output

print(mergeNumbers('31', '245'))
print(mergeNumbers('354', '3541'))

from queue import PriorityQueue
