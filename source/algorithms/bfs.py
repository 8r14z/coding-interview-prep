import collections

graph = {
    'a': ['b', 'c'],
    'b': ['a', 'c'],
    'c': ['b', 'a', 'd', 'e'],
    'd': ['c'],
    'e': ['c']}


def bfs(graph):
    reached = set()
    queue = collections.deque()
    queue.append('a')

    while queue:
        v = queue.popleft()
        if v not in reached:
            reached.add(v)
            queue += graph[v]
    return reached


print(bfs(graph))