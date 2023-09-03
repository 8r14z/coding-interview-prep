# https://leetcode.com/problems/find-median-from-data-stream/description/
import heapq
class MedianFinder:

    def __init__(self):
        self.max = []
        self.min = []

    def addNum(self, num: int) -> None:
        heapq.heappush(self.max, -num)
        heapq.heappush(self.min, -heapq.heappop(self.max))

        if len(self.max) < len(self.min): # ensure it's max queue justified. -> odd value is always on the max heap
            heapq.heappush(self.max, -heapq.heappop(self.min))

    def findMedian(self) -> float:
        if len(self.max) > len(self.min):
            return -self.max[0]
        else:
            return (self.min[0] - self.max[0])/2
        
