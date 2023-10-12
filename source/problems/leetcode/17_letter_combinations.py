# https://leetcode.com/problems/letter-combinations-of-a-phone-number
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        if len(digits) == 0:
            return []
            
        self.combi = ""
        self.result = []
        digitmap = {
            '2': 'abc',
            '3': 'def',
            '4': 'ghi',
            '5': 'jkl',
            '6': 'mno',
            '7': 'pqrs',
            '8': 'tuv',
            '9': 'wxyz'
        }
        def backtrack(i = 0):
            if i == len(digits):
                self.result.append(self.combi)
                return
            
            digit = digits[i]
            chars = digitmap[digit]
            for char in chars:
                self.combi += char
                backtrack(i+1)
                self.combi = self.combi[0:-1]

        backtrack()
        return self.result