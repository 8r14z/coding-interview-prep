# https://leetcode.com/problems/reconstruct-itinerary/
from collections import defaultdict
class Solution(object):
    def findItinerary(self, tickets):
    
        def build_graph(tickets):
            graph = defaultdict(list)
            for ticket in tickets:
                origin, dest = ticket
                graph[origin].append(dest)

            for origin in graph:
                graph[origin].sort(reverse=True)
            
            return graph
        
        def dfs(graph, source, order):
            while graph[source]:
                dest = graph[source].pop()
                dfs(graph, dest, order)
            
            order.append(source)
        
        graph = build_graph(tickets)
        output = []
        dfs(graph, 'JFK', output)
        return output[::-1]