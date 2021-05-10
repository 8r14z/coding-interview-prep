class Solution:
    def nextIndex(self, nums: [int], index: int) -> int:
        n = len(nums)
        next = index + nums[index]
        if next > n-1 or next < 0:
            return next % n
        else:
            return next

    def circularArrayLoop(self, nums: [int]) -> bool:
        n = len(nums)
    
        for i in range(n):
            curIdx = i
            visited = set()
            while curIdx not in visited:
                if (nums[i] > 0 and nums[curIdx] < 0) or (nums[i] < 0 and nums[curIdx] > 0):
                    break

                visited.add(curIdx)
                curIdx = self.nextIndex(nums, curIdx)

            if curIdx == i and len(visited) > 1:
                return True

        return False

print(Solution().circularArrayLoop([-2,-17,-1,-2,-2]))


