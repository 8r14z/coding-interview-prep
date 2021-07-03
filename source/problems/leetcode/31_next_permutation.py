# https://leetcode.com/problems/next-permutation/

class Solution:
    def nextPermutation(self, nums: List[int]) -> None:
        n = len(nums)
        
        i = n-2
        while i >= 0 and nums[i] >= nums[i+1]:
            i -= 1
         
        if i >= 0:
            j = n-1
            while j > i and nums[j] <= nums[i]:
                j -= 1
        
            nums[i], nums[j] = nums[j], nums[i]

        start = i + 1
        end = n - 1
        while start < end:
            nums[start], nums[end] = nums[end], nums[start]
            start += 1
            end -= 1
            
        