# https://leetcode.com/problems/exclusive-time-of-functions/

class Solution:
    def exclusiveTime(self, n: int, logs: List[str]) -> List[int]:
        stack = []
        ans = [0] * n
    
        for log in logs:
            function_id, status, cur_timestamp = log.split(':')
            function_id = int(function_id)
            cur_timestamp = int(cur_timestamp)
            
            if status == 'start':
                stack_frame = (function_id, cur_timestamp) 
                stack.append((stack_frame)
            else:
                start_index, start_timestamp = stack.pop()
                time = cur_timestamp - start_timestamp + 1
                ans[start_index] += cur_timestamp - start_timestamp + 1
                
                if stack:
                    last_index = stack[-1][0]
                    ans[last_index] -= time
                
        return ans
        