# https://leetcode.com/problems/maximum-length-of-repeated-subarray/

class Solution:
    def findLength(self, nums1: List[int], nums2: List[int]) -> int:
        n = len(nums1)
        m = len(nums2)
        
        if n < m:
            n,m = m,n
            nums1,nums2 = nums2,nums1
        
        previous = [0] * (n+1)
        current = [0] * (n+1)
        answer = 0
        for i in range(n):
            for j in range(m):
                if nums1[i] == nums2[j]:
                    current[j] = 1 + previous[j-1]
                else:
                    current[j] = 0
                    
                answer = max(answer, current[j])
            previous[:] = current
                
        return answer