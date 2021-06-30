# https://leetcode.com/problems/rotate-array/submissions/


class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        # dummy_arr = list(nums)
        n = len(nums)
        k = k % n
        if k == 0: 
            return 
        
        cur_index = 0
        cur_val = nums[cur_index]
        count = 0
        
        while count < n:
            start_index = cur_index
            
            while True:
                next_index = (cur_index+k) % n
                nums[next_index], cur_val = cur_val, nums[next_index]
                
                cur_index = next_index
                count += 1
                
                if cur_index == start_index:
                    break
                
            cur_index += 1
            cur_val = nums[cur_index]

    def rotate(self, nums: List[int], k: int) -> None:
        dummy_arr = list(nums)
        n = len(nums)
        k = k%n
        
        for i in range(n):
            nums[i] = dummy_arr[(n-k+i)%n]
            
            
        
            
        
        
                
                
        
        
        