# https://leetcode.com/problems/generate-parentheses/

# the intuition is to write open if open and close are equal
# to make a valid parentheses, there needs to be an open before close
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        def backtrack(string, open, close, strings):
            if open == 0 and close == 0:
                strings.append(string)
                return
                
            if open == close:
                backtrack(string + '(', open - 1, close, strings)
            elif open == 0:
                backtrack(string + ')', open, close - 1, strings)
            else:
                backtrack(string + '(', open - 1, close, strings)
                backtrack(string + ')', open, close - 1, strings)
        
        res = []
        backtrack('', n, n, res)
        return res