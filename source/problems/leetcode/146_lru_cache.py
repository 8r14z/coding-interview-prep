# https://leetcode.com/problems/lru-cache/
from collections import OrderedDict

class LRUCache:

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._cache = OrderedDict()

    def get(self, key: int) -> int:
        self._cache.
        if key in self._cache: 
            self._cache.move_to_end(key)
            return self._cache[key]
        else:
            return -1

    def put(self, key: int, value: int) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
            
        self._cache[key] = value
        self._cache.
        if len(self._cache) > self.capacity:
            self._cache.popitem(last = False)
            self._cache.pop()