# https://leetcode.com/problems/kth-largest-element-in-an-array/
class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        n = len(nums)
        if k > n: return -1
        
        def partition(left, right, arr) -> int:
            pivot = arr[right]
            pivot_index = left # save last index that > pivot
            
            for i in range(left,right):
                if arr[i] <= pivot:
                    arr[i], arr[pivot_index] = arr[pivot_index], arr[i]
                    pivot_index += 1
                    
            arr[right], arr[pivot_index] = arr[pivot_index], arr[right]
            
            return pivot_index
            
        left = 0
        right = n-1
        target = n-k
        
        while left <= right:
            pivot_index = partition(left, right, nums)
            if pivot_index == target:
                return nums[pivot_index]
            elif pivot_index > target:
                right = pivot_index - 1
            else:
                left = pivot_index + 1
                
        return -1
        
        