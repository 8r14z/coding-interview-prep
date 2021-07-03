# https://leetcode.com/problems/merge-intervals/

class Solution:
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        if not intervals: return []
        
        n = len(intervals)
        if n < 1: 
            return intervals
        intervals.sort(key=lambda interval: (interval[0], interval[1]))
        merged = []
        
        for i in range(n):
            if not merged or intervals[i][0] > merged[-1][1]:
                merged.append(intervals[i])
            else:
                merged[-1][1] = max(merged[-1][1], intervals[i][1])
            
        return merged
        