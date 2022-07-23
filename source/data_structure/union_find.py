# Union Find == Disjoint Sets

class UnionFind:
    # initialize the data structure that maps the node to its set ID
    def __init__(self):
        self.id = {}

    # find the Set ID of Node x
    def find(self, x):
        # get the value associated with key x, if it's not in the map return x
        y = self.id.get(x, x)
        # check if the current node is a Set ID node
        if y != x:
            # set the value to Set ID node of node y
            self.id[x] = y = self.find(y)
        return y

    # union two different sets setting one Set's parent to the other parent
    def union(self, x, y):
        self.id[self.find(x)] = self.find(y)
