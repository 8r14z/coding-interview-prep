class Solution:
    def minSwaps(self, s: str) -> int:
        oneCount = 0
        zeroCount = 0
        n = len(s)
        for c in s:
            if c == '0':
                zeroCount += 1
            else:
                oneCount += 1
                
        if abs(zeroCount - oneCount) > 1: return -1
        
        newS = ''
        count = 0
        if zeroCount > oneCount:
            # 0101...0
            for i in range(n):
                if (i % 2 == 0 and s[i] != '0') or (i % 2 == 1 and s[i] != '1'):
                     count += 1
        elif oneCount > zeroCount:
            # 1010
            for i in range(n):
                if (i % 2 == 0 and s[i] != '1') or (i % 2 == 1 and s[i] != '0'):
                     count += 1
                        
        else:
            even = s[0]
            odd = '1' if even == '0' else '0'
            for i in range(n):
                if (i % 2 == 0 and s[i] != even) or (i % 2 == 1 and s[i] != odd):
                    count+=1
            
                        
        return count//2
                    
                    
print(Solution().minSwaps("1110000000100001010100101010000101010101001000001110101000010111101100000111110001000111010111101100001100001001100101011110100011111100000000100011111011110111111011110111010100111101011111111101101100101010110000011110110100101111000100000001100000"))