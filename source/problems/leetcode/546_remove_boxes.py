# https://leetcode.com/problems/remove-boxes/

class Solution:
    def removeBoxes(self, boxes: List[int]) -> int:
        N = len(boxes)
        
        @lru_cache(None)
        def dp(l,r,count):
            if l > r: return 0
            elif l == r: return (count+1)**2
            
            m = l
            while m <= r and boxes[m] == boxes[l]:
                m += 1
            
            count = count + m - l
            ans = dp(m, r, 0) + count**2
            
            for i in range(m, r+1):
                if boxes[i] == boxes[l]:
                    ans = max(ans, dp(m,i-1,0) + dp(i,r, count))
            return ans
        
        return dp(0, N-1, 0)

# Intuition: try to count continuous intervals and merge them with intervals appearing later.
# E.g: 122232