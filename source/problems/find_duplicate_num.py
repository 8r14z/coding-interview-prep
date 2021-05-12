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

        # solution 3: binary search 1 -> n-1 
        # binary search the list of possible element
        # m = (1 + n-1)//2
        # iterate the nums array to see how many numbers are <= m => called k
        # if k > m -> find duplicate on 1 -> m
        # if k <= m -> binary search m+1 -> n-1
        # end of the loop, return l or r 

# https://www.youtube.com/watch?v=zQ6eAdtDeqg
