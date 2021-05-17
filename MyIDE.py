from itertools import permutations
class Solution:
    def rearrangeSticks(self, n: int, k: int) -> int:    
        pers = list(permutations(range(1, n+1)))
        count = 0
        for l in pers: 
            longest = l[0]
            c = 1
            for i in range(1, n):
                if l[i] > longest:
                    c += 1
                    longest = l[i]
            if c == k:
                count += 1
            
        return count

ret = Solution().rearrangeSticks(3, 2)
print(ret)