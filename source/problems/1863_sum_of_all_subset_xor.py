# https://leetcode.com/problems/sum-of-all-subset-xor-totals/submissions/

class Solution:
    def subsetXORSum(self, nums: [int]) -> int:
        bitMask = 0
        for num in nums:
            bitMask |= num
        return bitMask * (2**(len(nums)-1))
        
        
        
             