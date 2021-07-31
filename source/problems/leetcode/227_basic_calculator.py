# https://leetcode.com/problems/basic-calculator-ii/

class Solution:
    def calculate(self, s: str) -> int:
        stack = []
        cur_num = ''
        cur_op = '+'
        for i,c in enumerate(s):
            if c.isnumeric():
                cur_num += c
            
            if (i == len(s) - 1) or c in ['+', '-', '*', '/']:
                num = int(cur_num)
                if cur_op == '+':
                    stack.append(num)
                elif cur_op == '-':
                    stack.append(-num)
                elif cur_op == '*':
                    last_num = stack.pop()
                    stack.append(last_num * num)
                elif cur_op == '/':
                    tmp = stack.pop()
                    tmp = abs(tmp)//num if tmp >= 0 else -(abs(tmp)//num)
                    stack.append(tmp)
                    
                cur_op = c
                cur_num = ''
                
            
        
        answer = 0
        while stack:
            answer += stack.pop()
            
        return answer
                