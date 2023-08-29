// https://leetcode.com/problems/permutations/
class Solution {
    func backtrack(_ ans: inout [[Int]], _ permutation: inout [Int], _ nums: [Int]) {
        if permutation.count == nums.count {
            ans.append(permutation)
            return
        }

        for i in 0..<nums.count {
            guard !permutation.contains(nums[i]) else {
                continue
            }

            permutation.append(nums[i])
            backtrack(&ans, &permutation, nums)
            permutation.popLast()
        }
    }
    
    func permute(_ nums: [Int]) -> [[Int]] {
        var ans: [[Int]] = []
        var permutation: [Int] = []
        backtrack(&ans, &permutation, nums)
        return ans
    }
}