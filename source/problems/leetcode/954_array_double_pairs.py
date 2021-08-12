# https://leetcode.com/problems/array-of-doubled-pairs/
from collections import Counter
class Solution:
    def canReorderDoubled(self, arr: List[int]) -> bool:
        arr.sort(key=abs)
        counter = Counter(arr)
        
        for num in arr:
            if counter[num] == 0:
                continue
            if counter[2*num] == 0:
                return False
            
            counter[num] -= 1
            counter[2*num] -= 1
            
        return True
        
        
# sort increasing by abs
# -2 2 -4 4
# if the array is valid, there is always a pair of num and 2*num

