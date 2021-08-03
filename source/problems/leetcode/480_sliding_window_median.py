# https://leetcode.com/problems/sliding-window-median/

import bisect
class Solution:
    def medianSlidingWindow(self, nums: [int], k: int) -> [float]:
        
        i = 0
        n = len(nums)
        ans = []
        window = []
        
        for i in range(n):
            if i >= k:
                window.remove(nums[i-k])
                # O(k)
                
            bisect.insort(window, nums[i])
            
            # O(k)
            if i >= k-1:
                if k % 2:
                    ans.append(float(window[k//2]))
                else:
                    ans.append((window[k//2] + window[k//2 - 1])/2)
            
                
        return ans
    
# O(n*k)

# O(nlogk) by maintaining self-balance BST