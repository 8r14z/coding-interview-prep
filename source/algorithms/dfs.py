import collections

graph = {
    'a': ['b', 'c'],
    'b': ['a', 'c'],
    'c': ['b', 'a', 'd', 'e'],
    'd': ['c'],
    'e': ['c']}

def dfs(graph):
    reached = set()
    stack = collections.deque()
    stack.append('a')

    while stack:
        v = stack.pop()
        if v not in reached:
            reached.add(v)
            print(v)
            stack += graph[v]
    return reached