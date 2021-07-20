# https://leetcode.com/problems/maximum-area-of-a-piece-of-cake-after-horizontal-and-vertical-cuts/
class Solution:
    def maxArea(self, h: int, w: int, horizontalCuts: List[int], verticalCuts: List[int]) -> int:
        
        h_cuts = sorted(horizontalCuts)
        v_cuts = sorted(verticalCuts)
        
        max_h = 0
        prev_h = 0
        for i in h_cuts:
            max_h = max(max_h, i - prev_h)
            prev_h = i
            
        if h_cuts[-1] != h:
            max_h = max(max_h, h - prev_h)
            
        max_w = 0
        prev_v = 0
        for i in v_cuts:
            max_w = max(max_w, i - prev_v)
            prev_v = i
            
        if v_cuts[-1] != w:
            max_w = max(max_w, w - prev_v)
            
        return (max_h * max_w) % (10**9 + 7)
        