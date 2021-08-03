# https://leetcode.com/problems/subsets-ii/

class Solution:
    def subsetsWithDup(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        n = len(nums)
        ans = []
        
        def backtrack(nums, subset, index):
            ans.append(subset)
            for i in range(index, len(nums)):
                if i > index and nums[i] == nums[i-1]:
                    continue
                backtrack(nums, subset + [nums[i]], i+1)
        
        backtrack(nums, [], 0)
        return ans
    
# Time complexity dominate by backtrack: 2^n subset in total and take n for creating new array `subset + [nums[i]] => O(2^n * n)
# Space O(n) => max stack call