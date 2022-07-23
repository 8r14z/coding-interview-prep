# https://leetcode.com/problems/course-schedule/
class Solution:
    def canFinish(self, numCourses: int, prerequisites: [[int]]) -> bool:
        def build_graph(n, edges):
            graph = {}
            for edge in edges:
                q, p = edge[0], edge[1]
                if q in graph:
                    graph[q].append(p)
                else:
                    graph[q] = [p]
            return graph
        
        def found_cycle(start, graph, visited, finished):
            visited.add(start)
            if start in graph:
                for neighbor in graph[start]:
                    if neighbor not in visited:
                        if found_cycle(neighbor, graph, visited, finished):
                            return True
                    elif neighbor not in finished: # ancestor
                        return True
            finished.add(start)
            return False
                    
        graph = build_graph(numCourses, prerequisites)
        visited = set()
        finished = set()
        
        for i in range(numCourses):
            if i in visited:
                continue
            if found_cycle(i, graph, visited, finished):
                return False
            
        return True
print(Solution().canFinish(5, [[1,4],[2,4],[3,1],[3,2]]))