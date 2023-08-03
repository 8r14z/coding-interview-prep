class Node:
    def __init__(self):
        self.children = {}
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = Node()
    
    def add(self, word: str):
        cur = self.root
        
        for c in word:
            if c not in cur.children:
                cur.children[c] = Node()
            cur = cur.children[c]

        cur.is_end = True

    def search(self, word: str):
        cur = self.root

        for c in word: 
            if c not in cur.children:
                return False
            cur = cur.children[c]

        return cur.is_end

    def search_prefix(self, prefix: str):
        cur = self.root

        for c in prefix:
            if c not in cur.children[c]:
                return False
            cur = cur.children[c]

        return True

trie = Trie()
trie.add('brian')
trie.add('bro')
print(trie.search('brian1')) # False
print(trie.search('brian')) # True
print(trie.search_prefix('br')) # True
