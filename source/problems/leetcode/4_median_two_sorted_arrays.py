# https://leetcode.com/problems/median-of-two-sorted-arrays/

class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        median = 0
        i = j = 0
        
        n = len(nums1)
        m = len(nums2)
        
        if n > m:
            n,m = m,n
            nums1,nums2 = nums2,nums1
            
        low = 0
        high = n
        
        while low <= high:
            i = (low + high) // 2
            j = (n + m + 1) // 2 - i
            
            if i < n and j > 0 and nums2[j-1] > nums1[i]:
                low = i + 1
            elif i > 0 and j < m and nums2[j] < nums1[i-1]:
                high = i - 1
            else:
                if i == 0:
                    median = nums2[j-1]
                elif j == 0:
                    median = nums1[i-1]
                else:
                    median = max(nums1[i-1], nums2[j-1])
                break
        
        if (n+m) % 2:
            return median
        
        if i == n:
            return (median + nums2[j]) / 2
        
        if j == m:
            return (median + nums1[i]) / 2
        
        return (median + min(nums1[i], nums2[j])) / 2
        