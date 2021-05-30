# https://leetcode.com/problems/minimum-xor-sum-of-two-arrays/
# Generate all possible pair with mask
# Example: A = [1,0,3], B = [5,3,4]
# First run: mask: 0b111
# Mask first element of A[0] by 0b110, 0b101, 0b011
# Second run:
#   - maske second element of A[1] by 0b100, 0b010 (from 0b110)
#   - or maske second element of A[1] by 0b100, 0b001 (from 0b101)
#   - or 0b010, 0b001 (from 0b011)
# so on and so forth
# and then get min of all possble sums
# cache dp to reuse the computed result as there will be duplicates. ie. 0b100 vs 0b100 (look at above example)
from functools import cache

class Solution:
    def minimumXORSum(self, A, B):
        n = len(A)

        @cache
        def dfs(mask, i):
            if i == n: return 0
            
            ans = float('inf')
            for j in range(n):
                if mask & 1<<j:
                    ans = min(ans, dfs(mask^(1<<j), i+1) + (A[i]^B[j]))
                    
            return ans
                              
        return dfs((1<<n)-1, 0)