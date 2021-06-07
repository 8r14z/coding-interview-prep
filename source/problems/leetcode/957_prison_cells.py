# https://leetcode.com/problems/prison-cells-after-n-days/

class Solution:
    def prisonAfterNDays(self, cells: List[int], n: int) -> List[int]:
        ans = cells
        t = 14 if n%14 == 0 else n%14
        
        for t in range(t):
            tmp = [0] * 8
            for i in range(1, 7):
                if ans[i-1] == ans[i+1]:
                    tmp[i] = 1
            ans = tmp
        
        return ans