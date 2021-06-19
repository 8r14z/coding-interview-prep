# https://leetcode.com/problems/generate-parentheses/

# the intuition is to write open if open and close are equal
# to make a valid parentheses, there needs to be an open before close
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:
        def dfs(string, open, close, strings):
            if open == 0 and close == 0:
                strings.append(string)
                return

            if open == 0:
                dfs(string + ')', 0, close - 1, strings)
            elif open == close:
                dfs(string + '(', open - 1, close, strings)
            else:
                dfs(string + '(', open - 1, close, strings)
                dfs(string + ')', open, close - 1, strings)
        
        res = []
        dfs('', n, n, res)
        return res
        