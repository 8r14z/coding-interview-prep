# https://leetcode.com/problems/minimum-speed-to-arrive-on-time/
# Binary search speed from 1 to max speed of the problem, 10^7 1e7
class Solution:
    
    def calculateTime(self, dist: List[int], speed: int) -> float:
        time = 0.0
        n = len(dist)
        
        for i, d in enumerate(dist):
            if i == n-1:
                time += d/speed
            else:
                time += math.ceil(d/speed)
        
        return time
        
    def minSpeedOnTime(self, dist: List[int], hour: float) -> int:
        n = len(dist)
        if n < 1: return -1
        
        if hour <= n-1:
            return -1
        
        low = 1
        high = 10**7
        
        minSpeed = float('inf')
        
        while low <= high:
            speed = (high + low)//2
            
            time = self.calculateTime(dist, speed)
            
            if time > hour:
                low = speed + 1
            else:
                minSpeed = min(minSpeed, speed)
                high = speed - 1
                
            
        return -1 if minSpeed > 1e7 else minSpeed
        
            
        
        
    