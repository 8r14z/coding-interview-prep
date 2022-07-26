# https://leetcode.com/problems/course-schedule-ii/

class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        def build_graph(edges):
            graph = {}
            for edge in edges:
                q,p = edge
                if q in graph:
                    graph[q].append(p)
                else:
                    graph[q] = [p]
            return graph
        
        def found_cycle(start, graph, visited, finished, result):
            visited.add(start)
            if start in graph:
                for neighbor in graph[start]:
                    if neighbor not in visited:
                        if found_cycle(neighbor, graph, visited, finished, result):
                            return True
                    elif neighbor not in finished: # ancestor
                        return True
            finished.add(start)
            result.append(start)
            return False
                    
        graph = build_graph(prerequisites)
        visited = set()
        finished = set()
        result = []
        
        for i in range(numCourses):
            if i in visited:
                continue
            if found_cycle(i, graph, visited, finished, result):
                return []
            
        return result[::]
                    