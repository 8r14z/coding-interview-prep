# https://leetcode.com/problems/decode-string/
class Solution:
    def decodeString(self, s: str) -> str:       
        count_stack = []
        string_stack = []
        answer = ''
        
        for c in s:
            if c.isalpha():
                if len(string_stack) == 0:
                    answer += c
                else:
                    string_stack[-1] += c
            elif c.isdigit():
                if len(count_stack) == 0 or len(count_stack) == len(string_stack):
                    count_stack.append('')
                count_stack[-1] += c
            elif c == '[':
                string_stack.append('')
            else:
                repeated_string = string_stack.pop()
                count = int(count_stack.pop())
                tmp = ''
                
                for _ in range(count):
                    tmp += repeated_string
                    
                if len(string_stack) == 0:
                    answer += tmp
                else:
                    string_stack[-1] += tmp
        
        return answer