# https://leetcode.com/problems/stone-game/solution/

class Solution:
    def stoneGame(self, piles: List[int]) -> bool:
        @lru_cache(None)
        def helper(i,j,alex,lee):
            if i > j:
                return alex > lee
            return helper(i+1,j-1, alex + piles[i], lee + piles[j]) or helper(i+1,j-1, alex + piles[j], lee + piles[i])

        return helper(0,len(piles)-1, 0,0)