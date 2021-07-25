# https://leetcode.com/problems/trapping-rain-water/
class Solution:
    def trap(self, height: List[int]) -> int:
        n = len(height)
        
        if n < 1:
            return 0
        
        max_index = -1
        for i in range(n):
            if max_index == -1 or height[i] > height[max_index]:
                max_index = i
        answer = 0
        left_height = 0
        for i in range(max_index):
            if height[i] > left_height:
                left_height = height[i]
            else:
                answer += (left_height - height[i])
                
        right_height = 0
        for i in reversed(range(max_index+1, n)):
            if height[i] > right_height:
                right_height = height[i]
            else:
                answer += (right_height - height[i])
                
        return answer
                
            