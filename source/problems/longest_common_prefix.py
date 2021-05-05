class Solution:
    def longestCommonPrefix(self, strs: [str]) -> str:
        res = ''
        
        for i in range(1, len(strs)):
            prefix = self.commonPrefix(strs[i], strs[i-1])
            if res == '' or len(prefix) < len(res):
                res = prefix
        
        return res
    
    def commonPrefix(self, str1, str2):
        n = min(len(str1), len(str2))
        prefix = ''
        for i in range(n):
            if str1[i] != str2[i]:
                break
            prefix += str1[i]
        return prefix

print(Solution().longestCommonPrefix(['dog', 'racecard', 'card']))