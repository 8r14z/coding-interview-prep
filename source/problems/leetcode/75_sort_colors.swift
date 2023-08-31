// https://leetcode.com/problems/sort-colors/description/
class Solution {
    func sortColors(_ nums: inout [Int]) {
        var start = 0
        for i in start..<nums.count {
            if nums[i] == 0 {
                nums.swapAt(start, i)
                start += 1
            }
        }

        for i in start..<nums.count {
            if nums[i] == 1 {
                nums.swapAt(start, i)
                start += 1
            }
        }
    }
}

// New algo acquirred: https://en.wikipedia.org/wiki/Dutch_national_flag_problem
class Solution {
    func sortColors(_ nums: inout [Int]) {
        var low = 0 // keep track of 0's index 
        var mid = 0
        var high = nums.count-1

        while mid <= high {
            if nums[mid] == 0 {
                nums.swapAt(mid, low)
                low += 1
                mid += 1
            } else if nums[mid] == 1 {
                mid += 1
            } else {
                nums.swapAt(mid, high)
                high -= 1
            }
        }
    }
}