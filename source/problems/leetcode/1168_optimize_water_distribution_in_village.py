# https://leetcode.com/problems/optimize-water-distribution-in-a-village/

import heapq
# Prim + MinHeap
class Solution:
    def minCostToSupplyWater(self, n: int, wells: List[int], pipes: List[List[int]]) -> int:
        graph = defaultdict(list)
        
        for i, cost in enumerate(wells): 
            graph[0].append((cost, i+1))
        
        for pipe in pipes:
            a,b,cost = pipe
            graph[a].append((cost, b))
            graph[b].append((cost, a))
    
        houses = set([0])    
        heap = graph[0]
        heapq.heapify(heap)
        answer = 0
        
        while len(houses) < n + 1:
            cost, next_house = heapq.heappop(heap)
            if next_house not in houses:
                houses.add(next_house)
                answer += cost
                
                for node in graph[next_house]:
                    if node[1] not in houses:
                        heapq.heappush(heap, node)
        
        return answer