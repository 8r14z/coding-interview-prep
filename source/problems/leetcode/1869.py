class Solution:
    def checkZeroOnes(self, s: str) -> bool:
        max0 = 0
        max1 = 0
        
        prev = s[0]
        count = 1
        # 111000
        for i in range(1, len(s)):
            if s[i] == prev:
                count += 1
            else:
                if prev == '0':
                    max0 = max(max0, count)
                else:
                    max1 = max(max1, count)
                count = 1
                
            prev = s[i]
        
        if prev == '0':
            max0 = max(max0, count)
        else:
            max1 = max(max1, count)
            
        return max1 > max0
            