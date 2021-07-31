# https://leetcode.com/problems/add-strings/

class Solution:
    def addStrings(self, num1: str, num2: str) -> str:
        n = len(num1)
        m = len(num2)
        
        if n < m:
            n,m = m,n
            num1,num2 = num2,num1
            
        num1 = list(num1)
        num2 = list(num2)
        carrier = 0
        
        for i in reversed(range(n)):
            j = i - (n-m)
            res = ord(num1[i]) - ord('0') + carrier
            if j >= 0:
                res += (ord(num2[j]) - ord('0'))

            if res > 9:
                num1[i] = str(res%10)
                carrier = 1
            else:
                num1[i] = str(res)
                carrier = 0
                
        if carrier:
           num1.insert(0, '1')
        
        return ''.join(num1)
            