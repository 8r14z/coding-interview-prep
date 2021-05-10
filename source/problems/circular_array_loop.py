# https://leetcode.com/problems/circular-array-loop/

class Solution:
    # def circularArrayLoop(self, nums: [int]) -> bool:
    #     n = len(nums)
    
    #     for i in range(n):
    #         curIdx = i
    #         visited = set()
    #         while curIdx not in visited:
    #             if (nums[i] > 0 and nums[curIdx] < 0) or (nums[i] < 0 and nums[curIdx] > 0):
    #                 break

    #             visited.add(curIdx)
    #             next = self.next(nums, curIdx)
    #             if next == curIdx:
    #                 break

    #         if curIdx == i and len(visited) > 1:
    #             return True

    #     return False

    def next(self, nums: [int], index: int) -> int:
        n = len(nums)
        next = index + nums[index]
        if next > n-1 or next < 0:
            return next % n
        else:
            return next

    def sameSign(self, a: int, b: int) -> bool:
        return (a > 0 and b > 0) or (a < 0 and b < 0)

    def circularArrayLoop(self, nums: [int]) -> bool:
        n = len(nums)
        
        for i in range(n):
            fast = i
            slow = i
            origin = nums[i]
            if nums[i] == 0: continue
            while True:
                nextSlow = self.next(nums, slow)
                nextFast = self.next(nums, fast)

                if nextFast == fast or not self.sameSign(origin, nums[nextFast]):
                    break

                fast = nextFast
                nextFast = self.next(nums, nextFast)

                if nextSlow == slow or nextFast == fast: 
                    break
                if not self.sameSign(origin, nums[nextSlow]) or not self.sameSign(origin, nums[nextFast]):
                    break
                
                if nextFast == nextSlow:
                    return True
                
                fast = nextFast
                slow = nextSlow
            
            slow = i
            while self.sameSign(origin, nums[slow]):
                next = self.next(nums, slow)
                nums[slow] = 0
                slow = next

        return False
        

print(Solution().circularArrayLoop([-1,-2,-3,-4,-5]))


