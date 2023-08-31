// https://leetcode.com/problems/partition-equal-subset-sum/description/
// similar to knapsack un-repeated items
// [../ksnapsack.py]
class Solution {
    func canPartition(_ nums: [Int]) -> Bool {
        let sum = nums.reduce(0, +)
        if sum % 2 == 1 {
            return false
        }

        let subsetSum = sum / 2
        let n = nums.count
        // find subset that have sum = subsetSum
        var dp = Array(repeating: Array(repeating: false, count: subsetSum+1), count: n)
        for i in 0..<n {
            dp[i][0] = true
        }

        for i in 0..<n {
            for sum in 1...subsetSum {
                guard i > 0 else {
                    dp[i][sum] = sum == nums[i]
                    continue
                }

                dp[i][sum] = dp[i-1][sum]
                if sum >= nums[i] {
                    dp[i][sum] = dp[i][sum] || dp[i-1][sum-nums[i]]
                }
            }
        }

        return dp[n-1][subsetSum]
    }
}