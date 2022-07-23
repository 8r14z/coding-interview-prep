class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        if x not in id:
            return x
        # the routine to merge x to root
        root_x = self.find(self.parent[x])
        self.parent[x] = root_x
        return root_x

    # union set x to set y
    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            self.parent[root_x] = root_y