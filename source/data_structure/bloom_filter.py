# https://www.geeksforgeeks.org/bloom-filters-introduction-and-python-implementation/

import mmh3
import math
from bitarray import bitarray

class BloomFilter:
    def __init__(self, item_count, prob):
        self.item_count = item_count
        self.prob = prob
    