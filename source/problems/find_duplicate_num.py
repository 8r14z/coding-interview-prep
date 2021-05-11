# https://leetcode.com/problems/find-the-duplicate-number/

class Solution:
    def findDuplicate(self, nums: List[int]) -> int:
        # Solution 1: 2 poiters
        slow = fast = nums[0]
        while True:
            slow = nums[slow]
            fast = nums[nums[fast]]
            if slow == fast:
                break
                
        fast = nums[0]
        while fast != slow:
            fast = nums[fast]
            slow = nums[slow]
            
        return fast

        # solution 2: modify array
        # while nums[0] != nums[nums[0]]:
        #     nums[0], nums[nums[0]] = nums[nums[0]], nums[0]

        # return nums[0]
        
