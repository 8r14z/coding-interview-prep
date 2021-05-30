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